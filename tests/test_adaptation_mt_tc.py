"""User-run MT-TC objective and inference-loss checks; not agent-run preflight."""
import copy
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.adaptation_mt_tc.evaluate import load_study,validate_objective,compare_configs


class MTTCChecks(unittest.TestCase):
    def test_frozen_objective_cohort(self):
        study=load_study()
        self.assertEqual(study['variant'],'MT_TC_SE_adLIF')
        self.assertEqual(study['comparison_loss'],'supervised_cross_entropy_auxiliary_JS_excluded')
        self.assertEqual([r['selected_epoch'] for r in study['runs'] if r['model']==study['variant']],[232,249,226])
        self.assertEqual(study['samples'],1631)
        self.assertEqual(len(study['conditions']),6)

    def test_training_objective_parameters_are_explicit(self):
        study=load_study();run={'model':study['variant']}
        cfg=dict(training_objective=study['variant_training_objective'],temporal_consistency={'weight':.1,'window':5})
        validate_objective(cfg,run,study)
        for key,value in [('weight',.2),('window',10)]:
            changed=copy.deepcopy(cfg);changed['temporal_consistency'][key]=value
            with self.assertRaises(ValueError):validate_objective(changed,run,study)
        with self.assertRaises(ValueError):validate_objective(cfg,{'model':study['parent']},study)

    def test_common_configuration_guard_is_retained(self):
        parent=dict(l1={'cell':'mt'},l2={'cell':'mt'},lr=.01,evaluation_protocol='paper_classification_v1')
        variant=dict(parent,training_objective='mt_temporal_consistency_v1',evaluation_protocol='paper_classification_v1_mt_temporal_consistency_v1')
        compare_configs(parent,variant)
        with self.assertRaises(ValueError):compare_configs(parent,dict(variant,lr=.02))

    def test_frozen_prediction_loss_does_not_call_training_or_js(self):
        import torch
        from scripts.adaptation_analysis.pilot import infer
        class FakeModel:
            def __call__(self,x):return torch.tensor([[[2.,0.],[2.,0.]]]).expand(x.shape[0],-1,-1)
            def training_step(self,*args):raise AssertionError('Training called')
        class FakeData:
            def model_item(self,sample_id,values):
                return torch.zeros(2,1),torch.tensor(0,dtype=torch.long),torch.ones(2,dtype=torch.long)
        cfg=dict(dataset={'batch_size':1},loss_agg='sum_softmax_over_time',model_variant='MT_TC_SE_adLIF',random_seed=42,
                 training_objective='mt_temporal_consistency_v1',temporal_consistency={'weight':.1,'window':5})
        inputs=[(1,None,dict(sample_id=1,target=0))]
        with patch('functional.temporal_consistency.temporal_js_loss',side_effect=AssertionError('JS called')), \
             contextlib.redirect_stdout(io.StringIO()):
            one=infer(FakeModel(),cfg,FakeData(),inputs,torch.device('cpu'),'user test')
            cfg['temporal_consistency']['weight']=999
            two=infer(FakeModel(),cfg,FakeData(),inputs,torch.device('cpu'),'user test')
        self.assertEqual(one[0]['loss'],two[0]['loss'])
        self.assertEqual(one[0]['prediction'],0)


if __name__=='__main__':unittest.main()
