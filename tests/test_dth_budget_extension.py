"""Non-training checks for narrowly declared budget extensions."""
import copy
import unittest
from module.study_control import expected_training_budget


class BudgetExtensionTests(unittest.TestCase):
    def setUp(self):
        self.baseline={'datasets':{'SSC':{'n_epochs':40},'SHD':{'n_epochs':300},'ECG':{'n_epochs':400}}}
        self.spec={'continuation_manifest':'results/experiments/example/CONTINUATION_PLAN.json',
                   'datasets':{'SSC':{'training_budget_epochs':70,'original_schedule_epochs':40,
                                     'budget_extension_policy':'constant_plateau_budget_extension_v1'}}}
        self.cfg={'paper_baseline_dataset':'SSC','scheduler':'plateau','patience':9999,
                  'study_original_schedule_epochs':40,'study_budget_extension_policy':'constant_plateau_budget_extension_v1'}

    def test_declared_constant_plateau_extension(self):
        self.assertEqual(expected_training_budget(self.cfg,self.spec,self.baseline),70)

    def test_ordinary_protocol_retains_original_budget(self):
        self.assertEqual(expected_training_budget({'paper_baseline_dataset':'SSC'},{'datasets':{'SSC':{}}},self.baseline),40)
        self.assertEqual(expected_training_budget({'paper_baseline_dataset':'SHD'},{'datasets':{'SHD':{}}},self.baseline),300)

    def test_bad_or_undeclared_extensions_rejected(self):
        for key,value in [('scheduler','cosine'),('patience',10),('study_original_schedule_epochs',41),('study_budget_extension_policy','unchecked')]:
            cfg=copy.deepcopy(self.cfg);cfg[key]=value
            with self.assertRaises(ValueError):expected_training_budget(cfg,self.spec,self.baseline)
        for field,value in [('training_budget_epochs',True),('training_budget_epochs',30),('original_schedule_epochs',50)]:
            spec=copy.deepcopy(self.spec);spec['datasets']['SSC'][field]=value
            with self.assertRaises(ValueError):expected_training_budget(self.cfg,spec,self.baseline)
        spec=copy.deepcopy(self.spec);spec.pop('continuation_manifest')
        with self.assertRaises(ValueError):expected_training_budget(self.cfg,spec,self.baseline)
        with self.assertRaises(ValueError):expected_training_budget(self.cfg,{'datasets':{'SSC':{}}},self.baseline)

    def test_other_datasets_cannot_use_SSC_exception(self):
        for dataset,original in [('SHD',300),('ECG',400)]:
            spec=copy.deepcopy(self.spec);spec['datasets'][dataset]=dict(spec['datasets']['SSC'],training_budget_epochs=original+30,original_schedule_epochs=original)
            cfg=dict(self.cfg,paper_baseline_dataset=dataset,study_original_schedule_epochs=original)
            with self.assertRaises(ValueError):expected_training_budget(cfg,spec,self.baseline)


if __name__=='__main__':unittest.main()
