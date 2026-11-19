"""USER-RUN contracts for matched budgets, shared cohorts and source guards."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from project_paths import ROOT, local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_gradient_deletion.contracts import (
    GUIDED, RANDOM, atomic_csv, build_delta, control_seed, linear_prediction,
    load_primary, plan_controls, require_clean_checks, select_primary,
)
from scripts.adaptation_gradient_deletion.reporting import summarize
from scripts.adaptation_gradient_pilot.methods import array_hash, validate_delta

MODELS=['MT_SE_adLIF','AMT_SE_adLIF']
SPEC=dict(version='test_deletion',random_seed=20261116,deletion_fraction=.01,random_replicates=3)


class ControlContracts(unittest.TestCase):
    def test_primary_selection_is_fixed_repeat_zero(self):
        rows=[dict(model=m,sample_id=i,seed=42,layout='singleton_natural',repeat=r) for m in MODELS for i in range(40) for r in range(2)]
        selected=select_primary(rows,MODELS,list(range(40)))
        self.assertEqual(len(selected),80);self.assertTrue(all(r['repeat']==0 for r in selected.values()))

    def test_missing_or_duplicate_primary_fails(self):
        rows=[dict(model=m,sample_id=i,seed=42,layout='singleton_natural',repeat=0) for m in MODELS for i in range(40)]
        for bad in [rows[:-1],rows+[rows[0]]]:
            with self.assertRaises(ValueError):select_primary(bad,MODELS,list(range(40)))

    def test_exact_one_percent_budget_and_shared_plan(self):
        counts=np.array([[40,0],[0,60]],dtype=np.int64)
        budget,eligible,plan=plan_controls(counts,np.ones_like(counts),SPEC)
        self.assertEqual(budget,1);self.assertTrue(eligible)
        self.assertEqual(plan,[(GUIDED,0),(RANDOM,0),(RANDOM,1),(RANDOM,2)])

    def test_zero_event_guidance_keeps_three_random_controls(self):
        counts=np.array([[2,0],[0,0]],dtype=np.int64);gradient=np.array([[0.,1000.],[0.,0.]])
        budget,eligible,plan=plan_controls(counts,gradient,SPEC)
        self.assertEqual(budget,1);self.assertFalse(eligible);self.assertEqual(plan,[(RANDOM,i) for i in range(3)])

    def test_empty_input_has_zero_budget_and_no_fake_guidance(self):
        budget,eligible,plan=plan_controls(np.zeros((2,2),dtype=np.int64),np.ones((2,2)),SPEC)
        self.assertEqual(budget,0);self.assertFalse(eligible);self.assertEqual(len(plan),3)

    def test_guidance_uses_absolute_score_and_existing_cells(self):
        counts=np.array([[2,0],[0,3]],dtype=np.int64);g=np.array([[1.,999.],[999.,-4.]])
        delta=build_delta(counts,g,2,GUIDED,None)
        self.assertEqual(delta.tolist(),[[0,0],[0,-2]])

    def test_guidance_ties_use_stable_cell_order(self):
        counts=np.array([[2,3],[0,0]],dtype=np.int64)
        delta=build_delta(counts,np.ones((2,2)),3,GUIDED,None)
        self.assertEqual(delta.tolist(),[[-2,-1],[0,0]])

    def test_random_exact_event_budget_is_reproducible(self):
        counts=np.array([[2,3],[0,5]],dtype=np.int64)
        seed=control_seed(SPEC,3896,RANDOM,0)
        a=build_delta(counts,np.ones((2,2)),6,RANDOM,seed)
        b=build_delta(counts,np.ones((2,2)),6,RANDOM,seed)
        np.testing.assert_array_equal(a,b);validate_delta(counts,a,6)
        self.assertEqual(-a.sum(),6);self.assertTrue(np.all(counts+a>=0))
        self.assertNotEqual(seed,control_seed(SPEC,3896,RANDOM,1));self.assertIsNone(control_seed(SPEC,3896,GUIDED,0))

    def test_domain_violation_rejected(self):
        with self.assertRaises(ValueError):validate_delta(np.ones((2,2),dtype=np.int64),np.full((2,2),-2,dtype=np.int64),8)

    def test_own_gradient_linear_prediction_excludes_padding(self):
        delta=np.array([[-1,0],[0,-2]],dtype=np.int64)
        a=np.array([[1.,2.],[3.,4.],[999.,999.]],dtype=np.float32)
        self.assertEqual(linear_prediction(a,delta),-9.)
        self.assertEqual(linear_prediction(a*2,delta),-18.)


class SourceAndGateContracts(unittest.TestCase):
    def primary_fixture(self,path,repeat=0):
        backend=dict(matmul_precision='highest',matmul_allow_tf32=False)
        g=np.ones((2,140),dtype=np.float32)
        identity=dict(manifest_sha256='source',backend=backend,model=MODELS[0],sample_id=3896,seed=42,
            layout='singleton_natural',repeat=repeat)
        value=dict(model=MODELS[0],sample_id=3896,seed=42,layout='singleton_natural',repeat=repeat,stored_steps=2,
            gradient_sha256=array_hash(g),autograd_forward_check={'passed':True},strict_singleton_reference_check={'passed':True})
        np.savez_compressed(path,gradient=g,identity_json=json.dumps(identity),summary_json=json.dumps(value))
        item=dict(model=MODELS[0],sample_id=3896,sha256=pilot.sha256(path))
        return item,backend

    def test_primary_embedded_identity_is_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'g.npz';item,backend=self.primary_fixture(path)
            value,g=load_primary(path,item,'source',backend)
            self.assertEqual(value['repeat'],0);self.assertEqual(g.shape,(2,140))
            with self.assertRaises(ValueError):load_primary(path,item,'different',backend)

    def test_repeat_one_cannot_substitute_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'g.npz';item,backend=self.primary_fixture(path,repeat=1)
            with self.assertRaises(ValueError):load_primary(path,item,'source',backend)

    def test_all_eighty_clean_checks_gate_deletion(self):
        rows=[dict(model=m,sample_id=i,passed=True) for m in MODELS for i in range(40)]
        require_clean_checks(rows)
        for bad in [rows[:-1],rows[:-1]+[rows[0]],rows[:-1]+[dict(rows[-1],passed=False)]]:
            with self.assertRaises(ValueError):require_clean_checks(bad)

    def test_atomic_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'effects.csv';rows=[dict(model='MT,seed42',delta=-.125)]
            atomic_csv(path,rows)
            with path.open(newline='') as stream:value=list(csv.DictReader(stream))
            self.assertEqual(value,[{'model':'MT,seed42','delta':'-0.125'}])


def case(sid,guided=True):
    conditions=[]
    plan=[('clean',0,0.,0.)]+([(GUIDED,0,1.,2.)] if guided else [])
    plan += [(RANDOM,0,-1.,-1.),(RANDOM,1,0.,1.),(RANDOM,2,2.,0.)]
    for condition,rep,parent,variant in plan:
        conditions.append(dict(condition=condition,replicate=rep,budget=0 if condition=='clean' else 1,
            input_sha256=f'{sid}_{condition}_{rep}',evaluations=[dict(model=m,loss=4.+change,prediction=0,correct=1,
                true_probability=.5,actual_loss_change=change,surrogate_linear_prediction=-change)
                for m,change in zip(MODELS,[parent,variant])]))
    return dict(sample_id=sid,target=0,speaker=1,guidance_eligible=guided,conditions=conditions)


class ReportingAndEntryPointContracts(unittest.TestCase):
    def test_random_effects_average_within_same_eligible_record(self):
        s,paired,effects,contrasts=summarize([case(1),case(2,False)],MODELS)
        self.assertEqual(s['common_guidance_eligible_ids'],[1]);self.assertEqual(len(contrasts),2)
        self.assertEqual(len(paired),9);self.assertEqual(len(effects),18)
        self.assertEqual(s['groups'][0]['mean_abs_random_loss_change'],1.)
        self.assertEqual(s['groups'][0]['mean_guided_minus_random_abs_loss_change'],0.)
        self.assertAlmostEqual(contrasts[0]['mean_random_actual_loss_change'],1/3)
        self.assertEqual(s['n_training_seeds'],1)

    def test_zero_guidance_produces_undefined_comparison(self):
        s,_,_,contrasts=summarize([case(1,False)],MODELS)
        self.assertEqual(contrasts,[]);self.assertEqual(s['guided_eligible_records'],0)
        self.assertIsNone(s['groups'][0]['mean_abs_guided_loss_change'])

    def test_check_only_never_executes(self):
        from scripts.adaptation_gradient_deletion.run import main
        with patch('scripts.adaptation_gradient_deletion.run.prepare',return_value={}),patch('scripts.adaptation_gradient_deletion.runtime.execute') as execute:
            self.assertEqual(main(['--check-only']),0);execute.assert_not_called()

    def test_verify_only_never_prepares_or_executes(self):
        from scripts.adaptation_gradient_deletion.run import main
        with patch('scripts.adaptation_gradient_deletion.run.prepare') as prepare,patch('scripts.adaptation_gradient_deletion.runtime.execute') as execute,patch('scripts.adaptation_gradient_deletion.runtime.verify',return_value={'outputs':{}}):
            self.assertEqual(main(['--verify-only','--output-root','unused']),0)
            prepare.assert_not_called();execute.assert_not_called()

    def test_paths_resolve_against_moved_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout=Path(directory)/'moved';checkout.mkdir()
            name='results/experiments/example/gradients/seed_42/sample.npz'
            self.assertEqual(local_path(name,checkout),(checkout/name).resolve())
            self.assertEqual(portable_path(checkout/name,checkout),name)
        self.assertEqual(local_path('config/studies/adaptation_gradient_deletion_v1.yaml'),ROOT/'config/studies/adaptation_gradient_deletion_v1.yaml')


if __name__=='__main__':unittest.main()
