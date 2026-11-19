"""USER-RUN ONLY: analytical toy backward and diagnostic/cache contracts."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_pilot.methods import cached_array, state_hash
from scripts.adaptation_gradient_strict.methods import (
    gradient_comparison, input_gradient, require_anchor, score_check, validate_gradient,
)

SPEC=dict(score_atol=2e-5,score_rtol=1e-5,gradient_relative_l2_limit=.01,gradient_cosine_minimum=.999,
          anchor_accuracy_tolerance=1e-6,anchor_loss_tolerance=.001,burn_in_frames=1,profile_bins=2)


class StabilityContracts(unittest.TestCase):
    def test_score_bound_is_explicit_and_enforced(self):
        ref=dict(loss=2.,prediction=1)
        self.assertTrue(score_check(dict(loss=2.+3e-5,prediction=1),ref,SPEC)['passed'])
        self.assertFalse(score_check(dict(loss=2.+5e-5,prediction=1),ref,SPEC)['passed'])
        self.assertFalse(score_check(dict(loss=2.,prediction=9),ref,SPEC)['passed'])

    def test_identical_gradients(self):
        g=np.arange(12,dtype=np.float32).reshape(3,4)
        r=gradient_comparison(g,g,3,SPEC)
        self.assertTrue(r['passed']);self.assertEqual(r['symmetric_relative_l2'],0)
        self.assertAlmostEqual(r['cosine'],1.)

    def test_relative_metric_is_symmetric(self):
        a=np.ones((3,4));b=a*1.001
        x=gradient_comparison(a,b,3,SPEC);y=gradient_comparison(b,a,3,SPEC)
        self.assertAlmostEqual(x['symmetric_relative_l2'],y['symmetric_relative_l2'])
        self.assertTrue(x['passed'])

    def test_direction_flip_is_unstable(self):
        a=np.ones((3,4));r=gradient_comparison(a,-a,3,SPEC)
        self.assertFalse(r['passed']);self.assertAlmostEqual(r['cosine'],-1)

    def test_zero_maps_are_explicit(self):
        zero=np.zeros((3,4));r=gradient_comparison(zero,zero,3,SPEC)
        self.assertTrue(r['passed']);self.assertEqual(r['zero_status'],'both_zero');self.assertIsNone(r['cosine'])
        r=gradient_comparison(zero,np.ones((3,4)),3,SPEC)
        self.assertFalse(r['passed']);self.assertEqual(r['zero_status'],'one_zero');self.assertIsNone(r['cosine'])

    def test_padding_is_excluded_from_physical_repeat_metric(self):
        a=np.ones((5,4));b=a.copy();b[3:]=999
        self.assertTrue(gradient_comparison(a,b,3,SPEC)['passed'])
        self.assertFalse(gradient_comparison(a,b,5,SPEC)['passed'])

    def test_nonfinite_gradient_rejected(self):
        a=np.ones((3,4));a[0,0]=np.nan
        with self.assertRaises(ValueError):gradient_comparison(a,a,3,SPEC)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.stack([torch.full((140,),.01),torch.full((140,),.02)]))

    def forward(self,x):return x.sum(1)@self.weight.T


class DifferentiationContracts(unittest.TestCase):
    def inputs(self):
        counts=np.zeros((3,140),dtype=np.int64);counts[0,0]=2;counts[2,3]=1
        x=torch.zeros(1,5,140);x[0,:3]=torch.from_numpy(counts)
        batch=(x,torch.tensor([[-1,1]],dtype=torch.long),torch.tensor([[0,1,1,1,1]],dtype=torch.long))
        return counts,batch

    def fake_supervised(self,model,x,targets,blocks,cfg):
        self.assertEqual(targets.dtype,torch.int64)
        logits=model(x);loss=F.cross_entropy(logits,targets[:,1],reduction='none')
        return loss,logits

    def test_analytic_ce_input_gradient_and_frozen_state(self):
        counts,batch=self.inputs();model=Toy().requires_grad_(False).eval();before=state_hash(model)
        with patch('scripts.adaptation_gradient_strict.methods.supervised',side_effect=self.fake_supervised), \
             patch('scripts.adaptation_gradient_pilot.engine.supervised',side_effect=self.fake_supervised):
            with torch.no_grad():
                value,arrays=input_gradient(model,batch,{},torch.device('cpu'),counts,SPEC,before)
        probability=model(batch[0]).softmax(-1).detach()
        expected=((probability-torch.tensor([[0.,1.]]))@model.weight).repeat(5,1)
        torch.testing.assert_close(torch.from_numpy(arrays['gradient']),expected)
        self.assertTrue(value['autograd_forward_check']['passed']);self.assertEqual(state_hash(model),before)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.parameters()))
        validate_gradient(value,arrays,batch,counts,before)

    def test_inference_mode_rejected_before_forward(self):
        counts,batch=self.inputs();model=Toy().requires_grad_(False).eval()
        with torch.inference_mode(),patch('scripts.adaptation_gradient_strict.methods.ordinary') as ordinary:
            with self.assertRaises(ValueError):input_gradient(model,batch,{},torch.device('cpu'),counts,SPEC,state_hash(model))
            ordinary.assert_not_called()

    def test_state_mutation_detected(self):
        counts,batch=self.inputs();model=Toy().requires_grad_(False).eval();before=state_hash(model)
        def mutate(*args):
            with torch.no_grad():model.weight.add_(.001)
            return self.fake_supervised(*args)
        with patch('scripts.adaptation_gradient_strict.methods.supervised',side_effect=mutate), \
             patch('scripts.adaptation_gradient_pilot.engine.supervised',side_effect=self.fake_supervised):
            with self.assertRaisesRegex(ValueError,'Frozen model state changed'):
                input_gradient(model,batch,{},torch.device('cpu'),counts,SPEC,before)


class AnchorAndCacheContracts(unittest.TestCase):
    def test_same_mode_anchor_does_not_substitute_training_prediction(self):
        rows=[dict(sample_id=i,prediction=9,correct=0,loss=.87) for i in range(1631)]
        self.assertEqual(require_anchor(rows,rows,SPEC)['reference_mode'],'highest_matmul')
        changed=[dict(r) for r in rows];changed[0]['prediction']=1
        with self.assertRaises(ValueError):require_anchor(changed,rows,SPEC)

    def test_anchor_loss_mismatch_blocks_gradients(self):
        ref=[dict(sample_id=i,prediction=1,correct=1,loss=.1) for i in range(1631)]
        rows=[dict(r,loss=.12) for r in ref]
        with self.assertRaises(ValueError):require_anchor(rows,ref,SPEC)

    def test_orphan_array_resume_never_repeats_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'g.npz';contract=dict(mode='highest',repeat=0)
            cached_array(path,contract,lambda:({'finite':True},{'gradient':np.ones((3,140),dtype=np.float32)}),pilot.write_json)
            path.with_suffix('.json').unlink()
            with patch('builtins.print'):
                value,arrays,paths=cached_array(path,contract,lambda:self.fail('must not rerun backward'),pilot.write_json)
            self.assertTrue(value['finite']);self.assertEqual(len(paths),2);self.assertTrue(path.with_suffix('.json').exists())

    def test_missing_array_cannot_replace_published_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'g.npz'
            pilot.write_json(path.with_suffix('.json'),{'published':True})
            with self.assertRaises(ValueError):cached_array(path,{},lambda:self.fail('must not regenerate'),pilot.write_json)


class EntryPointContracts(unittest.TestCase):
    def test_check_only_does_not_execute(self):
        from scripts.adaptation_gradient_strict.run import main
        with patch('scripts.adaptation_gradient_strict.run.prepare',return_value={}),patch('scripts.adaptation_gradient_strict.runtime.execute') as execute:
            self.assertEqual(main(['--check-only']),0);execute.assert_not_called()

    def test_verify_only_does_not_prepare_or_execute(self):
        from scripts.adaptation_gradient_strict.run import main
        with patch('scripts.adaptation_gradient_strict.run.prepare') as prepare,patch('scripts.adaptation_gradient_strict.runtime.execute') as execute,patch('scripts.adaptation_gradient_strict.runtime.verify',return_value={'outputs':{}}):
            self.assertEqual(main(['--verify-only','--output-root','unused']),0)
            prepare.assert_not_called();execute.assert_not_called()

    def test_checkpoint_and_data_paths_are_checkout_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout=Path(directory)/'moved';checkout.mkdir()
            value='results/valid/MT_SE_adLIF/SHD/seed_42/ckpt/selected.ckpt'
            self.assertEqual(local_path(value,checkout),(checkout/value).resolve())
            self.assertEqual(portable_path(checkout/value,checkout),value)
        self.assertEqual(local_path('config/studies/adaptation_gradient_strict_v1.yaml'),ROOT/'config/studies/adaptation_gradient_strict_v1.yaml')


if __name__=='__main__':unittest.main()
