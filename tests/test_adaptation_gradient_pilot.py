"""USER-RUN ONLY: includes toy forward/backward checks; never training or checkpoint inference."""
from pathlib import Path
import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from project_paths import local_path, portable_path
from models.helpers import spike_grad_injection_function
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_pilot.engine import differentiate
from scripts.adaptation_gradient_pilot.methods import (
    array_hash, atomic_npz, cached_array, event_budget, exact_random_delta, gradient_summary,
    guided_delta, require_frozen, state_hash, validate_delta,
)
from scripts.adaptation_gradient_pilot.preflight import source_file, measurement_schema
from scripts.adaptation_gradient_pilot.reporting import summarize
from scripts.adaptation_gradient_pilot import run


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.eye(2))

    def forward(self,x):
        return torch.nn.functional.linear(x,self.weight)


class GradientPilotContracts(unittest.TestCase):
    def test_exact_toy_input_chain_rule_and_unchanged_weights(self):
        model=Toy().requires_grad_(False).eval();before=state_hash(model)
        x=torch.tensor([[[1.,0.]]]);targets=torch.tensor([[-1,0]],dtype=torch.long);blocks=torch.tensor([[1]])
        value,arrays=differentiate(model,(x,targets,blocks),dict(loss_agg='sum_softmax_over_time'),torch.device('cpu'),
            np.array([[1,0]],dtype=np.int64),dict(autograd_forward_loss_tolerance=1e-6,burn_in_frames=0,profile_bins=20),before)
        p=torch.softmax(x[0,0],dim=0);q=torch.softmax(p,dim=0)
        expected=(torch.diag(p)-torch.outer(p,p))@(q-torch.tensor([1.,0.]))
        self.assertTrue(np.allclose(arrays['gradient'][0],expected.numpy(),atol=1e-7))
        self.assertEqual(state_hash(model),before)
        self.assertTrue(all(v.grad is None and not v.requires_grad for v in model.parameters()))
        self.assertAlmostEqual(value['autograd_forward_loss_difference'],0.)

    def test_slayer_is_the_declared_surrogate_not_a_hard_spike_finite_difference(self):
        x=torch.tensor([-1.,0.,1.],requires_grad=True)
        z=spike_grad_injection_function(x,5.,.2)
        self.assertEqual(z.detach().tolist(),[0.,0.,1.])
        z.sum().backward()
        expected=.2*5/(2*torch.exp(x.detach().abs()*5))
        self.assertTrue(torch.allclose(x.grad,expected))
        self.assertGreater(float(x.grad[1]),0.)

    def test_inference_mode_is_rejected_for_backward(self):
        model=Toy().requires_grad_(False).eval()
        with torch.inference_mode(),self.assertRaises(ValueError):
            differentiate(model,None,{},torch.device('cpu'),None,{},state_hash(model))

    def test_model_mutation_is_detected(self):
        model=Toy().requires_grad_(False).eval();before=state_hash(model)
        with torch.no_grad():model.weight[0,0]+=1
        with self.assertRaises(ValueError):require_frozen(model,before)

    def test_burn_in_inputs_and_padding_are_separate(self):
        g=np.ones((8,2),dtype=np.float32);counts=np.ones((5,2),dtype=np.int64)
        r=gradient_summary(g,counts,burn_in=2,bins=20)
        self.assertEqual(r['physical_abs_gradient_sum'],10.)
        self.assertEqual(r['burn_in_input_abs_gradient_sum'],4.)
        self.assertEqual(r['synthetic_padding_abs_gradient_sum'],6.)
        self.assertAlmostEqual(sum(r['relative_time_abs_mass']),1.)
        self.assertAlmostEqual(sum(r['channel_abs_mass']),1.)

    def test_zero_gradients_are_not_normalized_or_fabricated(self):
        r=gradient_summary(np.zeros((8,2),dtype=np.float32),np.ones((5,2),dtype=np.int64))
        self.assertIsNone(r['relative_time_abs_mass']);self.assertIsNone(r['channel_abs_mass'])
        self.assertEqual(r['deletion_ranking_status'],'no_nonzero_gradient_on_existing_events')
        self.assertIsNone(guided_delta(np.ones((5,2),dtype=np.int64),np.zeros((5,2)),1))

    def test_event_budget_is_exact_and_handles_empty_activity(self):
        self.assertEqual(event_budget(np.array([[90,10]],dtype=np.int64),.01),1)
        self.assertEqual(event_budget(np.array([[1,0]],dtype=np.int64),.01),1)
        self.assertEqual(event_budget(np.zeros((4,2),dtype=np.int64),.01),0)

    def test_random_deletion_is_reproducible_and_count_bounded(self):
        counts=np.array([[2,1,0],[0,3,2]],dtype=np.int64)
        delta=exact_random_delta(counts,4,29)
        validate_delta(counts,delta,4)
        self.assertTrue(np.array_equal(delta,exact_random_delta(counts,4,29)))
        self.assertEqual(delta[0,2],0)
        self.assertTrue(np.all(counts+delta>=0))

    def test_guidance_uses_absolute_parent_gradient_with_fixed_ties(self):
        counts=np.array([[2,1,1]],dtype=np.int64);g=np.array([[-3.,3.,1.]])
        delta=guided_delta(counts,g,3)
        self.assertEqual(delta.tolist(),[[-2,-1,0]])
        validate_delta(counts,delta,3)
        with self.assertRaises(ValueError):guided_delta(counts,np.zeros_like(g),5)

    def test_nonnegative_integer_domain_is_required(self):
        with self.assertRaises(ValueError):event_budget(np.array([[.5]]),.01)
        with self.assertRaises(ValueError):event_budget(np.array([[-1]]),.01)
        with self.assertRaises(ValueError):validate_delta(np.array([[1]]),np.array([[-2]]),2)

    def test_array_identity_binds_shape_and_dtype(self):
        a=np.arange(6,dtype=np.float32)
        self.assertNotEqual(array_hash(a),array_hash(a.reshape(2,3)))
        self.assertNotEqual(array_hash(a),array_hash(a.astype(np.float64)))

    def test_published_array_without_sidecar_recovers_without_recomputation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cached.npz';identity=dict(scope='toy')
            value,arrays,_=cached_array(p,identity,lambda:(dict(value=3),dict(gradient=np.ones((2,2),dtype=np.float32))),pilot.write_json)
            original=p.read_bytes();p.with_suffix('.json').unlink()
            with patch('builtins.print'):
                result,restored,_=cached_array(p,identity,lambda:(_ for _ in ()).throw(AssertionError('recomputed')),pilot.write_json)
            self.assertEqual(result,value);self.assertEqual(p.read_bytes(),original)
            self.assertTrue(np.array_equal(restored['gradient'],arrays['gradient']))

    def test_missing_array_cannot_silently_replace_an_existing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'cached.npz';p.with_suffix('.json').write_text('{}')
            with self.assertRaises(ValueError):cached_array(p,{},lambda:({},{}),pilot.write_json)

    def test_array_publication_retries_only_sharing_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'result.npz';calls=[]
            def replace(a,b):
                calls.append(1)
                if len(calls)<3:
                    e=PermissionError('sharing');e.winerror=32;raise e
                os.replace(a,b)
            atomic_npz(p,dict(x=np.arange(3)),replace=replace,sleep=lambda _delay:None)
            with np.load(p,allow_pickle=False) as r:self.assertEqual(r['x'].tolist(),[0,1,2])
            self.assertEqual(len(calls),3)

    def test_guided_random_summary_uses_only_common_guidance_eligible_records(self):
        models=['parent','variant'];gradients={};cases=[]
        for i in [1,2]:
            for model in models:
                gradients[model,i]=dict(physical_abs_gradient_sum=1.,synthetic_padding_abs_gradient_sum=0.,
                    burn_in_input_abs_gradient_sum=0.,source_batch_loss_difference=0.,autograd_forward_loss_difference=0.,
                    clean=dict(loss=1.,prediction=0),original_batch_reference=dict(prediction=0))
            conditions=[]
            for condition,rep,change in [('clean',0,0.)]+[('exact_random_deletion',j,.1 if i==1 else 100.) for j in range(3)]+([('mt_abs_gradient_deletion',0,.3)] if i==1 else []):
                conditions.append(dict(condition=condition,replicate=rep,budget=1,input_sha256='hash',
                    evaluations=[dict(model=m,loss=1.+change,actual_loss_change=change,surrogate_linear_prediction=0.) for m in models]))
            cases.append(dict(sample_id=i,target=0,speaker=0,conditions=conditions))
        summary,_=summarize(gradients,cases,models,[1,2])
        group=summary['guided_vs_random'][0]
        self.assertEqual(group['common_eligible_record_ids'],[1])
        self.assertAlmostEqual(group['mean_abs_loss_change_random'],.1)
        self.assertAlmostEqual(group['mean_guided_minus_random_abs_change'],.2)
        self.assertFalse(summary['population_inference'])
        self.assertEqual(summary['n_training_seeds'],1)

    def test_snapshot_and_relocated_checkout_paths_are_contained(self):
        before=Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory).resolve()
            self.assertEqual(source_file(base,'raw/a.json'),base/'raw/a.json')
            with self.assertRaises(ValueError):source_file(base,'../escape.json')
            actual=local_path('config/dataset/shd.yaml',must_exist=True)
            try:
                os.chdir(base);self.assertEqual(local_path(portable_path(actual)),actual)
                self.assertEqual(local_path('data/SHDWrapper/shd_train.h5',root=base),base/'data/SHDWrapper/shd_train.h5')
            finally:os.chdir(before)

    def test_check_only_never_calls_forward_or_backward_execution(self):
        with patch.object(run,'prepare',return_value={}), \
             patch('scripts.adaptation_gradient_pilot.runtime.execute',side_effect=AssertionError('scientific execution')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(['--check-only']),0)


    def test_legacy_shd_measurement_has_explicit_schema_not_invented_status(self):
        cfg=dict(paper_baseline_dataset='SHD',dataset=dict(_target_='datasets.shd.SHDLDM'),model_variant='MT_SE_adLIF')
        run=dict(model='MT_SE_adLIF',seed=42,checkpoint='results/valid/MT_SE_adLIF/SHD/seed_42/best.ckpt',
                 checkpoint_sha256='weights',selected_epoch=3)
        cost=dict(version='shd_costs_v1',measurement_identity='a'*64,seed=42,model='MT_SE_adLIF',saved_config=cfg,
                  checkpoint=run['checkpoint'],checkpoint_sha256='weights',best_epoch=3,environment=dict(matmul_precision='high'))
        self.assertEqual(measurement_schema(cost,run,cfg),'legacy_shd_costs_v1_without_status_field')
        self.assertNotIn('status',cost)
        cost['version']='unknown'
        with self.assertRaises(ValueError):measurement_schema(cost,run,cfg)
        cost.update(status='failed',dataset='SHD')
        with self.assertRaises(ValueError):measurement_schema(cost,run,cfg)


if __name__=='__main__':unittest.main()
