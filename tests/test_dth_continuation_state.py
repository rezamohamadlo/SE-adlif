"""No training: test config/routing migration preserves full checkpoint state."""
import copy
import importlib.util
import unittest
import torch
from project_paths import ROOT

path=ROOT/'results/experiments/dth_complement_v1/extended_100_70_70_v1/ACTIONS.py'
spec=importlib.util.spec_from_file_location('audited_continuation_actions',path)
actions=importlib.util.module_from_spec(spec);spec.loader.exec_module(actions)


class ContinuationStateTests(unittest.TestCase):
    def config(self,dataset='SHD'):
        return dict(n_epochs={'SHD':300,'SSC':40}[dataset],lr=.01,scheduler='plateau',patience=9999,
                    dropout=.15,dataset={'validate_on':.2,'random_seed':42},logdir='old',stage_ceiling=30)

    def test_model_optimizer_scheduler_loop_and_selection_scores_preserved(self):
        old=ROOT/'results/experiments/continuation_test_old';new=ROOT/'results/experiments/continuation_test_new'
        original=dict(epoch=29,global_step=750,hyper_parameters={'cfg':self.config()},
            state_dict={'weight':torch.tensor([[1.,2.],[3.,4.]])},
            optimizer_states=[{'state':{0:{'step':torch.tensor(750.),'exp_avg':torch.tensor([.1,.2])}}}],
            lr_schedulers=[{'last_epoch':30,'best':.9}],loops={'fit_loop':{'epoch_progress':{'completed':30}}},
            callbacks={'ModelCheckpoint':{'monitor':'val_acc','dirpath':str(old/'ckpt'),
                'best_model_path':str(old/'ckpt/epoch=14-step=375.ckpt'),
                'last_model_path':str(old/'ckpt/last.ckpt'),'best_model_score':torch.tensor(.98),
                'best_k_models':{str(old/'ckpt/epoch=14-step=375.ckpt'):torch.tensor(.98)}}})
        before=actions.tree_hash(original)
        cfg=dict(self.config(),logdir='new',stage_ceiling=100)
        derived=actions.rebind_state(original,cfg,old,new,'SHD')
        self.assertEqual(actions.tree_hash(original),before)
        for key in ('state_dict','optimizer_states','lr_schedulers','loops'):
            self.assertEqual(actions.tree_hash(original[key]),actions.tree_hash(derived[key]))
        self.assertEqual(derived['epoch'],29);self.assertEqual(derived['global_step'],750)
        self.assertIn('continuation_test_new',derived['callbacks']['ModelCheckpoint']['best_model_path'])
        self.assertEqual(derived['hyper_parameters']['cfg']['stage_ceiling'],100)

    def test_math_changes_cannot_hide_as_metadata(self):
        old=self.config()
        for key,value in [('lr',.02),('dropout',.3),('dataset',{'validate_on':.1,'random_seed':42}),('n_epochs',100)]:
            new=copy.deepcopy(old);new[key]=value
            with self.assertRaises(AssertionError):actions.assert_math_equivalence(old,new,'SHD')

    def test_SSC_budget_change_requires_explicit_unchanged_plateau(self):
        old=self.config('SSC');new=dict(old,n_epochs=70,study_original_schedule_epochs=40,
                                     study_budget_extension_policy='constant_plateau_budget_extension_v1')
        actions.assert_math_equivalence(old,new,'SSC')
        new['scheduler']='cosine'
        with self.assertRaises(AssertionError):actions.assert_math_equivalence(old,new,'SSC')


if __name__=='__main__':unittest.main()
