"""User-launched mathematical/alignment contracts; never checkpoint inference or training."""
from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from project_paths import local_path, portable_path
from scripts.adaptation_ecg.data import ValidationData
from scripts.adaptation_ecg_extension.inputs import aligned_resample, ExtensionData, signed_features, time_map
from scripts.adaptation_ecg_extension.analysis import associations, design, pair_records
from scripts.adaptation_ecg_extension.preflight import source_file
from scripts.adaptation_ecg_extension.runtime import reused_rows
from scripts.adaptation_analysis import pilot

POLICY = dict(rank_relative_tolerance=1e-10, minimum_residual_df=5)


def feature_rows(n=30):
    rng = np.random.default_rng(340)
    return [dict(sample_id=i, fast_slow_contrast=float(rng.uniform(.1, 2)), autocorr_tau_steps=int(rng.integers(1, 30)),
        events_per_step=float(rng.uniform(.01, .4)), stored_steps=1301, class_fractions=[1., 0., 0., 0., 0., 0.],
        fast_slow_reason=None, autocorr_reason=None, autocorr_censored=False) for i in range(n)]


def synthetic_data():
    x = np.zeros((1, 5, 4), dtype=np.int16)
    x[0, 2, 0] = -1
    y = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)
    return ExtensionData(ValidationData(x, y, [0]), dict(deletion_probability=.1, selection_seed=34,
        time_scaling={'time_scale_0.9': .9, 'time_scale_1.1': 1.1}))


class ECGExtensionContracts(unittest.TestCase):
    def test_time_map_identity_and_frozen_lengths(self):
        self.assertTrue(np.array_equal(time_map(1301, 1.), np.arange(1301)))
        self.assertEqual(len(time_map(1301, .9)), 1171)
        self.assertEqual(len(time_map(1301, 1.1)), 1431)
        self.assertEqual(time_map(5, 1.1).tolist(), [0, 1, 2, 2, 3, 4])

    def test_events_labels_and_masks_share_exactly_one_map(self):
        x = np.array([[-1], [0], [1], [0], [-1]], dtype=np.float32)
        labels = np.arange(5)
        mask = np.array([True, False, True, True, False])
        mapped, y, valid, index = aligned_resample(x, labels, mask, 1.1)
        self.assertTrue(np.array_equal(mapped, x[index]))
        self.assertTrue(np.array_equal(y, labels[index]))
        self.assertTrue(np.array_equal(valid, mask[index]))
        self.assertTrue(np.isin(mapped, [-1, 0, 1]).all())

    def test_resampling_count_changes_are_not_hidden(self):
        data = synthetic_data()
        source = data.item(0, 'clean', 0)[0]
        mapped = data.item(0, 'time_scale_1.1', 0)[0]
        self.assertEqual(int(np.abs(source).sum()), 1)
        self.assertEqual(int(np.abs(mapped).sum()), 2)

    def test_mapped_collation_rebuilds_target_aligned_blocks(self):
        data = synthetic_data()
        _x, targets, blocks = data.batch([0], 'time_scale_1.1', 0)
        self.assertEqual(targets.tolist(), [[-1, 0, 1, 2, 2, 3, 4]])
        self.assertEqual(blocks.tolist(), [[1, 2, 3, 4, 5, 6]])
        self.assertEqual(data.item(0, 'clean', 0)[-1]['source_targets_sha256'],
                         data.item(0, 'time_scale_1.1', 0)[-1]['source_targets_sha256'])

    def test_activity_features_preserve_polarity_separately(self):
        x = np.zeros((25, 2), dtype=np.float32)
        x[::3, 0] = -1
        x[::5, 1] = 1
        labels, mask = np.zeros(25, dtype=int), np.ones(25, dtype=bool)
        f = signed_features(x, labels, mask, dict(short_steps=2, long_steps=4))
        reversed_f = signed_features(-x, labels, mask, dict(short_steps=2, long_steps=4))
        self.assertEqual(f['total_events'], 14)
        self.assertEqual(f['negative_events'], 9)
        self.assertEqual(f['positive_events'], 5)
        self.assertEqual(f['fast_slow_contrast'], reversed_f['fast_slow_contrast'])
        self.assertEqual(f['autocorr_tau_steps'], reversed_f['autocorr_tau_steps'])
        self.assertAlmostEqual(f['polarity_balance'], -reversed_f['polarity_balance'])
        self.assertEqual(f['time_unit'], 'model_step')

    def test_silent_and_constant_acf_are_explicitly_undefined(self):
        for x, reason in [(np.zeros((250, 4)), 'silent_activity'), (np.ones((250, 4)), 'constant_activity')]:
            f = signed_features(x, np.zeros(250, dtype=int), np.ones(250, dtype=bool), dict(short_steps=20, long_steps=200))
            self.assertIsNone(f['autocorr_tau_steps'])
            self.assertEqual(f['autocorr_reason'], reason)
            self.assertFalse(f['autocorr_censored'])

    def test_no_acf_crossing_is_censored_not_imputed(self):
        x = np.zeros((1000, 1))
        x[500:] = 1
        f = signed_features(x, np.zeros(1000, dtype=int), np.ones(1000, dtype=bool), dict(short_steps=2, long_steps=20))
        self.assertIsNone(f['autocorr_tau_steps'])
        self.assertTrue(f['autocorr_censored'])

    def test_descriptive_design_drops_constant_duration_with_reason(self):
        x, keep, meta = design(feature_rows(), POLICY)
        self.assertEqual(meta['status'], 'eligible')
        self.assertEqual(len(keep), 30)
        self.assertEqual(x.shape, (30, 4))
        self.assertTrue(any(r['name'] == 'log1p_stored_steps' for r in meta['dropped_controls']))

    def test_descriptive_coefficients_and_seed_count(self):
        features = feature_rows()
        x, _keep, _meta = design(features, POLICY)
        seeds = [42, 123, 456]
        rows = []
        for j, seed in enumerate(seeds):
            y = x @ np.array([.1, 1.+j, -.5, .2])
            rows.extend(dict(sample_id=i, seed=seed, condition='clean', loss_benefit=float(y[i])) for i in range(30))
        r = associations(features, rows, seeds, POLICY)
        self.assertEqual(r['status'], 'descriptive_analysis_complete')
        self.assertAlmostEqual(r['feature_summary']['fast_slow_contrast']['mean'], 2., places=8)
        self.assertAlmostEqual(r['feature_summary']['fast_slow_contrast']['sample_sd_across_training_seeds'], 1., places=8)
        self.assertEqual(r['n_training_seeds'], 3)
        self.assertEqual(len(r['leave_one_record_out']), 30)
        self.assertIsNone(r['intervals'])
        self.assertIsNone(r['p_values'])
        self.assertFalse(r['inference'])

    def test_missing_features_are_explicit_complete_case_exclusions(self):
        rows = feature_rows()
        rows[0].update(autocorr_tau_steps=None, autocorr_reason='no_crossing_within_horizon', autocorr_censored=True)
        _x, keep, meta = design(rows, POLICY)
        self.assertEqual(len(keep), 29)
        self.assertEqual(meta['exclusions'][0]['sample_id'], 0)
        self.assertTrue(meta['exclusions'][0]['autocorr_censored'])
        self.assertEqual(len(rows), 30)

    def test_constant_primary_or_rank_failure_never_selects_a_replacement_model(self):
        rows = feature_rows()
        for row in rows: row['autocorr_tau_steps'] = 3
        x, _keep, meta = design(rows, POLICY)
        self.assertIsNone(x)
        self.assertTrue(meta['reason'].startswith('constant_primary_feature'))
        rows = feature_rows()
        for row in rows: row['autocorr_tau_steps'] = row['fast_slow_contrast']*2
        x, _keep, meta = design(rows, POLICY)
        self.assertIsNone(x)
        self.assertEqual(meta['reason'], 'rank_or_residual_df_guard')

    def test_residual_degrees_of_freedom_guard(self):
        x, _keep, meta = design(feature_rows(5), POLICY)
        self.assertIsNone(x)
        self.assertEqual(meta['reason'], 'rank_or_residual_df_guard')

    def test_mapped_pairing_allows_changed_targets_but_requires_shared_source(self):
        data = synthetic_data()
        records = []
        for condition, model in [('time_scale_1.1', 'parent'), ('time_scale_1.1', 'variant'),
                                 ('clean', 'parent'), ('clean', 'variant')]:
            identity = data.item(0, condition, 0)[-1]
            records.append(dict(identity, model=model, seed=42, valid_steps=6 if condition != 'clean' else 5,
                                loss=1. if model == 'parent' else .5, accuracy=.5))
        f = dict(data.item(0, 'time_scale_1.1', 0)[-1], total_events=2, source_total_events=1,
                 event_count_change=1, fast_slow_contrast=0., autocorr_tau_steps=None, autocorr_censored=True)
        features = {('time_scale_1.1', 0): {0: f}}
        rows = pair_records(*[[r] for r in records], features)
        self.assertEqual(rows[0]['event_count_change'], 1)
        self.assertEqual(rows[0]['benefit_change_from_clean'], 0.)
        damaged = deepcopy(records)
        damaged[1]['targets_sha256'] = 'wrong'
        with self.assertRaises(ValueError): pair_records(*[[r] for r in damaged], features)

    def test_reused_prediction_identity_cannot_be_rewritten_silently(self):
        data = synthetic_data()
        identity = data.item(0, 'clean', 0)[-1]
        row = dict(identity, model='parent', seed=42, valid_steps=5, loss_sum=5., loss=1., correct_steps=5,
                   accuracy=1., predictions=[0, 1, 2, 3, 4])
        # Old source lacks only the newly added metadata fields.
        for key in ('mask_sha256', 'time_map_sha256', 'source_targets_sha256'): row.pop(key)
        run = dict(model='parent', seed=42, checkpoint_sha256='weights')
        record = dict(identity=dict(manifest_sha256='source', checkpoint_sha256='weights', condition='clean', replicate=0),
                      value=[row], value_sha256=pilot.digest([row]))
        self.assertEqual(len(reused_rows(record, data, run, 'clean', 0, 'source')), 1)
        record['value'][0]['targets_sha256'] = 'wrong'
        record['value_sha256'] = pilot.digest(record['value'])
        with self.assertRaises(ValueError): reused_rows(record, data, run, 'clean', 0, 'source')

    def test_snapshot_paths_are_contained_and_checkout_independent(self):
        before = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            spec = dict(source=dict(root='results/experiments/example'))
            self.assertEqual(source_file(root, 'results/experiments/example/raw/a.json', spec), root/'raw/a.json')
            with self.assertRaises(ValueError): source_file(root, 'results/experiments/example/../../escape.json', spec)
            path = local_path('config/dataset/ecg.yaml', must_exist=True)
            try:
                os.chdir(root)
                self.assertEqual(local_path(portable_path(path)), path)
                self.assertEqual(local_path('data/QTDB/QTDB_train.mat', root=root), root/'data/QTDB/QTDB_train.mat')
            finally:
                os.chdir(before)


if __name__ == '__main__':
    unittest.main()
