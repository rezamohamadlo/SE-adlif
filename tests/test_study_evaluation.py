"""Frozen-plan guards only; no test dataset access or model training/evaluation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from omegaconf import OmegaConf
from project_paths import ROOT,portable_path
from test_checkpoint import sha256,validate_study_evaluation


class StudyEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'best.ckpt';self.path.write_bytes(b'trusted-test-fixture')
        self.receipt=Path(self.temp.name)/'RESULT.json'
        self.key=portable_path(self.path)
        self.cfg=OmegaConf.create(dict(model_variant='FIR_SE_adLIF',random_seed=42,paper_baseline_dataset='SHD'))
        self.entry=dict(sha256=sha256(self.path),stage_receipt=str(self.receipt),model_source_sha256={'source':'unchanged'})
        self.plan=dict(status='frozen',purpose='exploratory',test_selects_ideas=False,checkpoints={self.key:self.entry})
        self.record=dict(best_checkpoint=self.key,best_checkpoint_sha256=self.entry['sha256'],selected_epoch_index=63,
                         model='FIR_SE_adLIF',seed=42,dataset='SHD',completed_epochs=75,full_budget=300,preliminary=True)
        self.receipt.write_text(json.dumps(self.record))
        self.output=ROOT/'results/experiments/complementary_mechanisms_v1/evaluations/unit-fixture'
        self.source=patch('module.study_control.forward_identity',return_value={'source':'unchanged'})
        self.source.start();self.addCleanup(self.source.stop)

    def evaluate(self,plan=None,path=None,output=None):
        return validate_study_evaluation(plan or self.plan,path or self.path,dict(epoch=63),self.cfg,output or self.output)

    def test_exploratory_receipt_preserves_true_budget_and_selection(self):
        evidence=self.evaluate()
        self.assertTrue(evidence['preliminary']);self.assertEqual(evidence['completed_training_epochs'],75)
        self.assertEqual(evidence['selected_epoch_index'],63);self.assertEqual(evidence['full_budget'],300)

    def test_speaker_development_checkpoints_cannot_be_official_test_evidence(self):
        self.cfg.complementary_study='complementary_speaker_dev_v2'
        with self.assertRaises(ValueError):self.evaluate()

    def test_unfrozen_and_test_selected_plans_are_rejected(self):
        for key,value in [('status','draft'),('test_selects_ideas',True),('purpose','development')]:
            p=copy.deepcopy(self.plan);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.evaluate(plan=p)

    def test_exact_checkpoint_hash_and_implementation_required(self):
        for key,value in [('sha256','wrong'),('model_source_sha256',{'source':'changed'})]:
            p=copy.deepcopy(self.plan);p['checkpoints'][self.key][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.evaluate(plan=p)

    def test_last_or_other_checkpoint_cannot_replace_selected_best(self):
        self.record['best_checkpoint']='another.ckpt';self.receipt.write_text(json.dumps(self.record))
        with self.assertRaises(ValueError):self.evaluate()

    def test_training_output_cannot_be_contaminated_by_test_evidence(self):
        with self.assertRaises(ValueError):self.evaluate(output=ROOT/'results/experiments/complementary_mechanisms_v1/A001')

    def test_final_requires_frozen_four_arm_plan(self):
        p=copy.deepcopy(self.plan);p['purpose']='final'
        with self.assertRaises(ValueError):self.evaluate(plan=p)
        p.update(arms=['Baseline','A','B','A+B'],architecture_and_settings_frozen=True)
        with self.assertRaises(ValueError):self.evaluate(plan=p)
        p.update(datasets=['SHD','SSC','ECG'],training_seeds=[42,123,456])
        with self.assertRaises(ValueError):self.evaluate(plan=p)

    def test_complete_final_cohort_and_full_budget_are_verified(self):
        p=copy.deepcopy(self.plan)
        p.update(purpose='final',arms=['Baseline','A','B','A+B'],architecture_and_settings_frozen=True,
                 datasets=['SHD','SSC','ECG'],training_seeds=[42,123,456],checkpoints={})
        for arm in p['arms']:
            for dataset in p['datasets']:
                for seed in p['training_seeds']:
                    own=(arm=='A' and dataset=='SHD' and seed==42)
                    path=self.path if own else Path(self.temp.name)/f'{arm}_{dataset}_{seed}.ckpt'
                    receipt=self.receipt if own else path.with_suffix('.json')
                    if not own:path.write_bytes(b'trusted-test-fixture')
                    key=portable_path(path);budget={'SHD':300,'SSC':40,'ECG':400}[dataset]
                    record=dict(self.record,best_checkpoint=key,dataset=dataset,seed=seed,
                                status='stage_complete',completed_epochs=budget,full_budget=budget,preliminary=False)
                    receipt.write_text(json.dumps(record))
                    p['checkpoints'][key]=dict(self.entry,stage_receipt=str(receipt),arm=arm)
        self.assertEqual(self.evaluate(plan=p)['purpose'],'final')
        del p['checkpoints'][next(key for key in p['checkpoints'] if key!=self.key)]
        with self.assertRaises(ValueError):self.evaluate(plan=p)


if __name__=='__main__':unittest.main()
