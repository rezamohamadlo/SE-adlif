"""Scientific protocol guards only: synthetic metadata, no model/data/training."""
import copy
import unittest
from module.final_config_guard import validate_frozen_configurations


class FinalConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(frozen_configurations={},
                         mechanism_difference_paths=['model_variant', 'input_gain', 'training_objective'])
        self.cohort = []
        for dataset in ('SHD', 'SSC', 'ECG'):
            self.plan['frozen_configurations'][dataset] = {}
            for arm in ('Baseline', 'A', 'B', 'A+B'):
                cfg = dict(random_seed=42, paper_baseline_dataset=dataset,
                           dataset=dict(validate_on=.2, random_seed=42),
                           lr=.01, batch_size=256, n_epochs={'SHD':300,'SSC':40,'ECG':400}[dataset],
                           model_variant='GAIN' if arm in ('A','A+B') else 'SE',
                           input_gain=arm in ('A','A+B'),
                           training_objective='smooth' if arm in ('B','A+B') else 'classification',
                           paper_baseline_version='fixture', init_ckpt_path=None)
                self.plan['frozen_configurations'][dataset][arm] = cfg
                for seed in (42,123,456):
                    own=copy.deepcopy(cfg);own.update(random_seed=seed,logdir=f'results/{arm}/{dataset}/{seed}')
                    self.cohort.append((arm,dataset,seed,own))

    def check(self):
        return validate_frozen_configurations(self.plan,self.cohort)

    def test_complete_matched_four_arm_replication(self):
        self.assertEqual(self.check()['verified_configurations'],36)

    def annotate_objectives(self):
        for dataset,arms in self.plan['frozen_configurations'].items():
            for arm,cfg in arms.items():
                cfg['evaluation_protocol']='matched'+('_smooth' if arm in ('B','A+B') else '')
        for arm,dataset,seed,cfg in self.cohort:
            cfg['evaluation_protocol']=self.plan['frozen_configurations'][dataset][arm]['evaluation_protocol']

    def test_explicit_objective_annotation_preserves_common_protocol(self):
        self.annotate_objectives()
        self.assertEqual(self.check()['verified_configurations'],36)

    def test_objective_annotation_cannot_hide_a_different_base_protocol(self):
        self.annotate_objectives()
        self.plan['frozen_configurations']['SHD']['B']['evaluation_protocol']='other_smooth'
        for arm,dataset,seed,cfg in self.cohort:
            if arm=='B' and dataset=='SHD':cfg['evaluation_protocol']='other_smooth'
        with self.assertRaisesRegex(ValueError,'outside'):self.check()

    def test_full_frozen_protocol_annotation_still_required_per_checkpoint(self):
        self.annotate_objectives()
        next(cfg for arm,dataset,seed,cfg in self.cohort if arm=='B')['evaluation_protocol']='matched'
        with self.assertRaisesRegex(ValueError,'frozen arm'):self.check()

    def test_another_cohort_checkpoint_cannot_change_lr(self):
        self.cohort[-1][3]['lr']=.002
        with self.assertRaisesRegex(ValueError,'frozen arm'):self.check()

    def test_declaring_lr_as_mechanism_cannot_waive_matching(self):
        self.plan['mechanism_difference_paths'].append('lr')
        with self.assertRaisesRegex(ValueError,'waive'):self.check()

    def test_frozen_split_mismatch_is_rejected_even_if_every_seed_matches_it(self):
        self.plan['frozen_configurations']['SHD']['A']['dataset']['random_seed']=7
        with self.assertRaisesRegex(ValueError,'outside'):self.check()

    def test_speaker_holdout_cannot_be_relabelled_canonical(self):
        for arm in self.plan['frozen_configurations']['SHD']:
            self.plan['frozen_configurations']['SHD'][arm]['dataset']['validate_on']='speaker_holdout'
        for _,dataset,_,cfg in self.cohort:
            if dataset=='SHD':cfg['dataset']['validate_on']='speaker_holdout'
        with self.assertRaisesRegex(ValueError,'Canonical'):self.check()

    def test_duplicate_arm_implementations_are_rejected(self):
        f=self.plan['frozen_configurations']['SSC'];f['A']=copy.deepcopy(f['Baseline'])
        with self.assertRaisesRegex(ValueError,'distinct'):self.check()

    def test_different_labels_or_deviation_notes_do_not_create_four_mechanisms(self):
        f=self.plan['frozen_configurations']['SSC']
        for arm in f:
            f[arm]=dict(copy.deepcopy(f['Baseline']),model_variant=arm,
                        paper_baseline_deviations=['Renamed '+arm])
        with self.assertRaisesRegex(ValueError,'distinct'):self.check()

    def test_nonstring_mechanism_paths_are_rejected_cleanly(self):
        self.plan['mechanism_difference_paths'].append({'lr':True})
        with self.assertRaisesRegex(ValueError,'explicit mechanism'):self.check()

    def test_missing_and_duplicate_cohorts_are_rejected(self):
        self.cohort.pop()
        with self.assertRaisesRegex(ValueError,'incomplete'):self.check()
        self.cohort.append(self.cohort[0])
        with self.assertRaisesRegex(ValueError,'duplicate'):self.check()

    def test_weights_only_initialization_is_not_fresh_training(self):
        self.cohort[0][3]['init_ckpt_path']='results/previous/best.ckpt'
        with self.assertRaisesRegex(ValueError,'weights-only'):self.check()

    def test_actual_training_seed_must_match_cohort(self):
        self.cohort[1][3]['random_seed']=42
        with self.assertRaisesRegex(ValueError,'seed'):self.check()

    def test_unplanned_dropout_or_preprocessing_changes_rejected(self):
        for key,value in [('dropout',.4),('unregistered_preprocess',{'warp':True})]:
            with self.subTest(key=key):
                changed=copy.deepcopy(self.cohort);changed[0][3][key]=value
                with self.assertRaisesRegex(ValueError,'frozen arm'):
                    validate_frozen_configurations(self.plan,changed)

    def test_declared_mechanism_cannot_waive_shared_dropout_surrogate_or_time_step(self):
        for key in ('dropout','alpha','dt','accumulate_grad_batches','gradient_clip_algorithm'):
            with self.subTest(key=key):
                plan=copy.deepcopy(self.plan);cohort=copy.deepcopy(self.cohort)
                plan['mechanism_difference_paths'].append(key)
                for dataset,arms in plan['frozen_configurations'].items():
                    for arm,cfg in arms.items():
                        cfg[key]='changed' if arm=='A' else 'shared'
                for arm,dataset,seed,cfg in cohort:
                    cfg[key]=plan['frozen_configurations'][dataset][arm][key]
                with self.assertRaisesRegex(ValueError,'waive'):
                    validate_frozen_configurations(plan,cohort)

    def rdrop_fixture(self):
        plan=copy.deepcopy(self.plan);cohort=copy.deepcopy(self.cohort)
        plan['mechanism_difference_paths']=['model_variant','training_objective',
            'label_smoothing_epsilon','dropout_consistency_alpha']
        objectives={'Baseline':'classification','A':'classification_dropout_consistency_v1',
            'B':'classification_label_smoothing_v1','A+B':'classification_dropout_consistency_v1'}
        for dataset,arms in plan['frozen_configurations'].items():
            for arm,cfg in arms.items():
                cfg.pop('input_gain');cfg['model_variant']='synthetic_'+arm
                cfg.update(training_objective=objectives[arm],
                    label_smoothing_epsilon=.1 if arm in ('B','A+B') else 0,
                    dropout_consistency_alpha=1 if arm in ('A','A+B') else 0,
                    dropout=.15,alpha=5.,dt=1.,gradient_clip_val=1.5)
                cfg['evaluation_protocol']='canonical_fixture'+(
                    '_'+objectives[arm] if objectives[arm]!='classification' else '')
        for arm,dataset,seed,cfg in cohort:
            own=copy.deepcopy(plan['frozen_configurations'][dataset][arm])
            own.update(random_seed=seed,logdir=cfg['logdir']);cfg.clear();cfg.update(own)
        return plan,cohort

    def test_synthetic_two_objective_four_arm_complete_cohort_allowed(self):
        plan,cohort=self.rdrop_fixture()
        self.assertEqual(validate_frozen_configurations(plan,cohort)['verified_configurations'],36)

    def test_synthetic_rdrop_weight_or_smoothing_cannot_change_after_freeze(self):
        for key in ('dropout_consistency_alpha','label_smoothing_epsilon'):
            plan,cohort=self.rdrop_fixture()
            next(cfg for arm,dataset,seed,cfg in cohort if arm=='A+B')[key]=.5
            with self.assertRaisesRegex(ValueError,'frozen arm'):
                validate_frozen_configurations(plan,cohort)


if __name__=='__main__':unittest.main()
