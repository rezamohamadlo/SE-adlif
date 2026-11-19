"""User-launched pure contracts; no training, checkpoint forward or real-data diagnostics."""
from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from project_paths import local_path, portable_path
from scripts.adaptation_analysis import pilot
from scripts.adaptation_ssc_windows.metrics import METRICS, anchor_row, windows, validate_windows
from scripts.adaptation_ssc_windows.preflight import read_cached, source_file
from scripts.adaptation_ssc_windows.reporting import summarize
from scripts.adaptation_ssc_windows.runtime import validate_chunk
from scripts.adaptation_ssc_windows import run


def probability(z):
    e = np.exp(z-z.max(1, keepdims=True))
    return e/e.sum(1, keepdims=True)


def observation():
    z = np.tile([4., -4.], (300, 1))
    return dict(sample_id=0, target=0, speaker='speaker', input_sha256='hash', original_support_frames=30,
        stored_frames=300, model='MT_SE_adLIF', seed=42, prediction=0, correct=1, loss=.2,
        condition='clean', replicate=0, minimum_padding_frames=270, batch_padding_frames=0,
        windows=windows(z, probability(z), 0, 30, [5, 10, 20]))


class SSCWindowContracts(unittest.TestCase):
    def test_alternating_evidence_and_window_ce(self):
        z = np.r_[np.zeros((10, 2)), np.tile([1000., -1000.], (5, 1)), np.tile([-1000., 1000.], (5, 1))]
        row = windows(z, probability(z), 0, 20, [5])[0]
        self.assertAlmostEqual(row['adjacent_js_nats'], np.log(2))
        self.assertEqual(row['true_class_concentration'], 1.)
        self.assertEqual(row['window_accuracy'], .5)
        self.assertEqual(row['top_class_switch_fraction'], 1.)
        self.assertAlmostEqual(row['window_mean_supervised_ce'], 1000.)
        validate_windows([row], dict(window_frames=[5]), 0)

    def test_stable_wrong_outputs_are_not_useful_evidence(self):
        z = np.tile([-4., 4.], (30, 1))
        row = windows(z, probability(z), 0, 30, [5])[0]
        self.assertEqual(row['adjacent_js_nats'], 0.)
        self.assertEqual(row['window_accuracy'], 0.)
        self.assertEqual(row['window_supervised_accuracy'], 0.)
        self.assertGreater(row['window_mean_supervised_ce'], 8.)
        self.assertLess(row['true_class_mean_probability'], .001)

    def test_probability_and_membrane_window_accuracies_are_distinct(self):
        z = np.tile(np.array([[20., 0.], [-2., 0.], [-2., 0.]]), (2, 1))
        row = windows(z, probability(z), 0, 6, [3], burn_in=0)[0]
        self.assertEqual(row['window_accuracy'], 0.)
        self.assertEqual(row['window_supervised_accuracy'], 1.)
        self.assertAlmostEqual(row['window_mean_supervised_ce'], np.log1p(np.exp(-16/3)))

    def test_burn_in_and_synthetic_padding_do_not_enter_descriptors(self):
        a = np.tile([4., -4.], (80, 1))
        b = a.copy()
        b[:10] = [-400., 400.]
        b[30:] = [-400., 400.]
        for left, right in zip(windows(a, probability(a), 0, 30, [5, 10]),
                               windows(b, probability(b), 0, 30, [5, 10])):
            for metric in METRICS: self.assertAlmostEqual(left[metric], right[metric])

    def test_incomplete_tail_is_dropped_and_reported(self):
        z = np.tile([4., -4.], (31, 1))
        z[-1] = [-400., 400.]
        row = windows(z, probability(z), 0, 31, [10])[0]
        self.assertEqual(row['dropped_tail_frames'], 1)
        self.assertEqual(row['valid_windows'], 2)
        self.assertEqual(row['window_supervised_accuracy'], 1.)

    def test_short_support_is_explicitly_undefined(self):
        z = np.zeros((300, 2))
        row = windows(z, probability(z), 0, 19, [5])[0]
        self.assertEqual(row['valid_windows'], 1)
        self.assertTrue(all(row[k] is None for k in METRICS))
        self.assertEqual(row['undefined_reason'], 'fewer_than_two_complete_windows')

    def test_changed_ordinary_prediction_blocks_the_anchor(self):
        row = observation()
        expected = deepcopy(row)
        self.assertEqual(anchor_row(row, expected), 0.)
        row['prediction'] = 1
        with self.assertRaises(ValueError): anchor_row(row, expected)

    def test_cached_chunk_checks_support_identity_and_coverage(self):
        row = observation()
        spec = dict(window_frames=[5, 10, 20], burn_in_frames=10)
        self.assertEqual(validate_chunk([row], [0], dict(model=row['model'], seed=42), {0: row}, spec), [0.])
        changed = deepcopy(row)
        changed['windows'][0]['valid_windows'] += 1
        with self.assertRaises(ValueError):
            validate_chunk([changed], [0], dict(model=row['model'], seed=42), {0: row}, spec)
        with self.assertRaises(ValueError):
            validate_chunk([row], [0, 1], dict(model=row['model'], seed=42), {0: row}, spec)

    def test_paired_summary_counts_training_seeds_and_finite_pairs(self):
        values = {}
        for seed, difference in [(42, .01), (123, .02), (456, .03)]:
            a = np.zeros((3, 2, len(METRICS)))
            b = np.full_like(a, difference)
            a[0, 1, 1] = np.nan
            values['MT_SE_adLIF', seed] = a
            values['MT_TC_SE_adLIF', seed] = b
        r = summarize(values, [42, 123, 456], [5, 10, 20])
        selected = next(v for v in r['across_seed_differences'] if v['metric'] == METRICS[0] and v['window_frames'] == 5)
        self.assertEqual(selected['n_training_seeds'], 3)
        self.assertAlmostEqual(selected['mean'], .02)
        self.assertAlmostEqual(selected['sample_sd'], .01)
        selected = next(v for v in r['paired_per_seed_differences'] if v['metric'] == METRICS[1] and v['window_frames'] == 5)
        self.assertEqual(selected['n_finite_pairs'], 1)

    def test_cache_digest_damage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'cache.json'
            p.write_text(json.dumps(dict(value=[1], value_sha256=pilot.digest([2]))))
            with self.assertRaises(ValueError): read_cached(p)

    def test_source_paths_remain_contained_and_portable(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory).resolve()
            self.assertEqual(source_file(p, 'raw/clean.json'), p/'raw/clean.json')
            with self.assertRaises(ValueError): source_file(p, '../outside.json')
            existing = local_path('config/dataset/ssc.yaml', must_exist=True)
            self.assertEqual(local_path(portable_path(existing)), existing)
            self.assertEqual(local_path('data/SSCWrapper/ssc_valid.h5', root=p), p/'data/SSCWrapper/ssc_valid.h5')

    def test_check_only_never_calls_execution(self):
        with patch.object(run, 'prepare', return_value={}), \
             patch('scripts.adaptation_ssc_windows.runtime.execute', side_effect=AssertionError('experiment ran')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run.main(['--check-only']), 0)


if __name__ == '__main__':
    unittest.main()
