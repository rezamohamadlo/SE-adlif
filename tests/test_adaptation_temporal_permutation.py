"""USER-RUN ONLY: pure transformation/contracts; no model or experiment execution."""
import unittest
from unittest.mock import patch

import numpy as np

from scripts.adaptation_temporal_permutation.transform import permute, derive_seed
from scripts.adaptation_temporal_permutation.run import digest, output_path, summary


class PermutationTests(unittest.TestCase):
    def test_whole_windows_shared_channels_dtype_and_counts(self):
        for dtype in (np.int32, np.int64, np.float32, np.float64):
            x = np.arange(144).reshape(48, 3).astype(dtype)
            y, audit = permute(x, 5, 77, burn=3, support=46)
            self.assertEqual(y.dtype, x.dtype)
            np.testing.assert_array_equal(y.sum(0), x.sum(0))
            np.testing.assert_array_equal(y[:3], x[:3])
            np.testing.assert_array_equal(y[43:], x[43:])
            order = audit['regions'][0]['order']
            self.assertNotEqual(order, list(range(8)))
            for destination, source in enumerate(order):
                np.testing.assert_array_equal(y[3 + 5 * destination:8 + 5 * destination],
                                             x[3 + 5 * source:8 + 5 * source])
            np.testing.assert_array_equal(x, np.arange(144).reshape(48, 3).astype(dtype))

    def test_deterministic_and_seed_contract(self):
        x = np.arange(160).reshape(80, 2)
        s = derive_seed(20261119, 'SHD', 27, 5, 2)
        a, am = permute(x, 5, s)
        b, bm = permute(x, 5, s)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(am, bm)
        self.assertNotEqual(s, derive_seed(20261119, 'SSC', 27, 5, 2))
        self.assertNotEqual(s, derive_seed(20261119, 'SHD', 27, 10, 2))
        self.assertNotEqual(s, derive_seed(20261119, 'SHD', 27, 5, 1))

    def test_zero_short_empty_and_burn_longer_than_support(self):
        for n in (0, 1, 9, 10, 30):
            x = np.zeros((n, 3), dtype=np.float32)
            y, m = permute(x, 5, 9, burn=10)
            np.testing.assert_array_equal(x, y)
            self.assertFalse(m['effective'])
            self.assertEqual(m['status'], 'ineffective' if n >= 20 else 'ineligible')

    def test_binary_dtype_and_nonidentity_two_windows(self):
        x = np.zeros((10, 2), dtype=bool)
        x[:5] = True
        y, m = permute(x, 5, 3)
        self.assertEqual(y.dtype, x.dtype)
        self.assertEqual(m['regions'][0]['order'], [1, 0])
        np.testing.assert_array_equal(y[:5], x[5:])

    def test_label_boundaries_and_signed_polarities(self):
        x = np.tile(np.array([-1, 0, 1], dtype=np.float32), (34, 1))
        x[::3] *= -1
        labels = np.array([0] * 13 + [1] * 11 + [2] * 10)
        copy = labels.copy()
        y, m = permute(x, 5, 3, labels=labels, signed=True, burn=2)
        np.testing.assert_array_equal(labels, copy)
        for sign in (-1, 1):
            np.testing.assert_array_equal((y == sign).sum(0), (x == sign).sum(0))
        for region in m['regions']:
            lo, hi = region['start'], region['stop']
            self.assertEqual(len(set(labels[lo:hi])), 1)
            np.testing.assert_array_equal(y[lo:hi].sum(0), x[lo:hi].sum(0))
            np.testing.assert_array_equal(y[region['tail_start']:hi], x[region['tail_start']:hi])
        self.assertTrue(np.isin(y, [-1, 0, 1]).all())

    def test_invalid_inputs(self):
        x = np.zeros((20, 2))
        for kw in (dict(width=0), dict(width=True), dict(seed=-1), dict(burn=-1),
                   dict(support=21), dict(support=2.5), dict(labels=np.zeros(19, dtype=int))):
            args = dict(width=5, seed=1, **{})
            args.update(kw)
            with self.assertRaises(ValueError):
                permute(x, **args)
        for bad in (np.full((20, 2), np.nan), np.full((20, 2), .1), np.full((20, 2), -1)):
            with self.assertRaises(ValueError):
                permute(bad, 5, 1)
        with self.assertRaises(ValueError):
            permute(np.full((20, 2), 2), 5, 1, signed=True)


class WorkflowTests(unittest.TestCase):
    def test_frozen_source_identity_mismatch_is_not_bypassed(self):
        from scripts.adaptation_temporal_permutation.source import FrozenSources, normalized_hash
        loader = object.__new__(FrozenSources)
        loader.read = lambda path: b'value = 1\r\n'
        with self.assertRaises(ValueError):
            loader.authenticate({'missing.py': normalized_hash(b'value = 2\n')})

    def test_output_root_is_scoped_and_cwd_independent(self):
        from project_paths import ROOT
        self.assertEqual(output_path('results/experiments/adaptation_temporal_permutation_pilot_v1'),
                         ROOT / 'results/experiments/adaptation_temporal_permutation_pilot_v1')
        for path in ('results/valid/unsafe', 'results/experiments/other',
                     'results/experiments/adaptation_temporal_permutation_x/nested'):
            with self.assertRaises(ValueError):
                output_path(path)

    def test_summary_replicates_are_not_training_seeds(self):
        rows = []
        for seed, value in zip([42, 123, 456], [1., 2., 3.]):
            for sid in (1, 2):
                for rep in range(3):
                    scores = {side: dict(loss=value, accuracy=.5) for side in
                              ('clean_parent', 'clean_variant', 'permuted_parent', 'permuted_variant')}
                    rows.append(dict(dataset='SHD', pair='amt_mt', width=5, seed=seed, sample_id=sid,
                                     delta_benefit=value, **scores))
        result = summary(rows, dict(training_seeds=[42, 123, 456], permutation_replicates=3))[0]
        self.assertEqual(result['n_training_seeds'], 3)
        self.assertEqual(result['mean_delta_benefit'], 2.)
        self.assertEqual(result['sample_sd'], 1.)
        with self.assertRaises(ValueError):
            summary(rows[:-1], dict(training_seeds=[42, 123, 456], permutation_replicates=3))

    def test_check_only_does_not_call_execution_or_tests(self):
        from scripts.adaptation_temporal_permutation import run
        fake = type('Frozen', (), {'install': lambda self: None})
        with patch.object(run, 'FrozenSources', return_value=fake()), \
             patch.object(run, 'prepare', return_value=(None, {'cohorts': {'SHD': [1]}, 'blocked': {}}, {})), \
             patch.object(run, 'execute', side_effect=AssertionError('must not execute')), \
             patch.object(run.subprocess, 'run', side_effect=AssertionError('must not run tests')):
            self.assertEqual(run.main(['--check-only']), 0)

    def test_cache_tamper_rejected(self):
        from scripts.adaptation_temporal_permutation.run import Store
        import tempfile
        from pathlib import Path
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Exercise the cache guard without invoking a model or constructing a study.
            store = object.__new__(Store)
            store.root, store.manifest = root, {'v': 1}
            path = root / 'item.json'
            path.write_text(json.dumps(dict(identity=dict(manifest_sha256=digest(store.manifest), key=1),
                value=[1], value_sha256='wrong')), encoding='utf-8')
            with self.assertRaises(ValueError):
                store.cached('item.json', {'key': 1}, lambda: self.fail('tampered cache recomputed'))


if __name__ == '__main__':
    unittest.main()
