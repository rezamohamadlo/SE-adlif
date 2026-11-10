"""Pure count/feature checks: no models, dataset iteration, optimizers or fit."""
import json
import unittest

import numpy as np

from scripts.adaptation_analysis.features import measure_features
from scripts.adaptation_analysis.perturbations import thin_counts, warp_counts


class AdaptationFeatureTests(unittest.TestCase):
    def test_physical_support_ignores_only_explicit_zero_padding(self):
        x = np.zeros((120, 3), dtype=np.float32)
        x[:80:4, 0] = 2
        original = x.copy()
        a = measure_features(x[:80], support_frames=80)
        b = measure_features(x, support_frames=80)
        for key in ('total_spikes', 'mean_rate_hz', 'duration_ms',
                    'fast_slow_contrast', 'autocorr_tau_ms', 'active_bin_fraction'):
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(b['provided_frames'], 120)
        self.assertEqual(b['total_spikes'], 40)
        self.assertEqual(b['duration_ms'], 320)
        self.assertEqual(b['mean_rate_hz'], 125)
        np.testing.assert_array_equal(x, original)
        with self.assertRaises(ValueError):
            measure_features(x, support_frames=70)

    def test_constant_rate_contrast_zero_and_acf_undefined(self):
        result = measure_features(np.ones((100, 2), dtype=int), support_frames=100)
        self.assertEqual(result['fast_slow_contrast'], 0)
        self.assertTrue(result['fast_slow_valid'])
        self.assertIsNone(result['autocorr_tau_ms'])
        self.assertEqual(result['autocorr_reason'], 'constant_activity')
        self.assertEqual(result['active_channel_fraction'], 1)

    def test_burst_has_greater_contrast_than_constant_equal_count_input(self):
        constant = np.ones((200, 2), dtype=np.float32)
        burst = np.zeros_like(constant)
        burst[::20] = 20
        a, b = measure_features(constant), measure_features(burst)
        self.assertEqual(a['total_spikes'], b['total_spikes'])
        self.assertGreater(b['fast_slow_contrast'], a['fast_slow_contrast'])
        self.assertTrue(b['autocorr_valid'])

    def test_known_alternating_acf_crosses_after_one_bin(self):
        x = np.tile([[0], [2]], (80, 1))
        result = measure_features(x)
        self.assertEqual(result['autocorr_tau_ms'], 4)

    def test_no_acf_crossing_is_censored_instead_of_invented(self):
        x = np.r_[np.zeros(500), np.ones(500)].reshape(-1, 1)
        result = measure_features(x)
        self.assertIsNone(result['autocorr_tau_ms'])
        self.assertTrue(result['autocorr_censored'])
        self.assertEqual(result['autocorr_horizon_ms'], 200)

    def test_silent_short_and_empty_support_are_serializable(self):
        cases = [measure_features(np.zeros((100, 2))),
                 measure_features(np.ones((10, 2))),
                 measure_features(np.zeros((10, 2)), support_frames=0)]
        self.assertEqual(cases[0]['fast_slow_reason'], 'silent_input')
        self.assertEqual(cases[1]['fast_slow_reason'], 'insufficient_complete_windows')
        self.assertEqual(cases[2]['autocorr_reason'], 'empty_support')
        self.assertIsNone(cases[2]['mean_rate_hz'])
        for result in cases:
            json.dumps(result, allow_nan=False)

    def test_invalid_count_arrays_support_and_windows_are_rejected(self):
        for x in (np.zeros(3), np.zeros((0, 2)), [[-1]], [[np.nan]],
                  [[np.inf]], [[0.5]], [[1+0j]], [['1']]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                measure_features(x)
        x = np.ones((100, 2))
        for support in (-1, 101, 2.5, True):
            with self.subTest(support=support), self.assertRaises(ValueError):
                measure_features(x, support_frames=support)
        for kwargs in ({'bin_ms':0}, {'bin_ms':np.nan}, {'short_ms':21},
                       {'short_ms':200}, {'long_ms':-1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                measure_features(x, **kwargs)


class AdaptationPerturbationTests(unittest.TestCase):
    def test_thinning_identity_total_deletion_and_determinism(self):
        x = np.full((100, 4), 10, dtype=np.float32)
        original = x.copy()
        identity = thin_counts(x, 0, 4)
        np.testing.assert_array_equal(identity, x)
        self.assertEqual(identity.dtype, x.dtype)
        self.assertIsNot(identity, x)
        np.testing.assert_array_equal(thin_counts(x, 1, 4), np.zeros_like(x))
        a, b = thin_counts(x, .2, 123), thin_counts(x, .2, 123)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all((a >= 0) & (a <= x)))
        self.assertTrue(np.any((a > 0) & (a < x)))  # Individual events, not whole bins.
        np.testing.assert_array_equal(x, original)

    def test_thinning_expected_density_with_fixed_large_sample(self):
        x = np.full((5000, 3), 20, dtype=np.int32)
        out = thin_counts(x, .25, 42)
        self.assertAlmostEqual(float(out.sum()) / float(x.sum()), .75, delta=.005)
        self.assertEqual(out.dtype, x.dtype)

    def test_warp_identity_and_exact_mapping(self):
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
        identity = warp_counts(x, 1)
        np.testing.assert_array_equal(identity, x)
        self.assertEqual(identity.dtype, x.dtype)
        self.assertIsNot(identity, x)
        np.testing.assert_array_equal(warp_counts(x, .5), [[4, 6], [12, 14]])
        np.testing.assert_array_equal(warp_counts(x, 2),
                                      [[1, 2], [0, 0], [3, 4], [0, 0],
                                       [5, 6], [0, 0], [7, 8], [0, 0]])

    def test_warp_channel_counts_length_and_input_preserved(self):
        x = np.random.default_rng(42).integers(0, 7, size=(251, 4)).astype(np.float32)
        original = x.copy()
        for scale in (.01, .9, 1.1, 2.3):
            with self.subTest(scale=scale):
                out = warp_counts(x, scale)
                self.assertEqual(out.shape, (int(np.ceil(len(x)*scale)), x.shape[1]))
                np.testing.assert_array_equal(out.sum(axis=0), x.sum(axis=0))
                self.assertTrue(np.all(out == np.floor(out)))
        np.testing.assert_array_equal(x, original)

    def test_transform_invalid_parameters_and_counts_are_rejected(self):
        x = np.ones((10, 2))
        for p in (-.1, 1.1, np.nan, np.inf, True):
            with self.subTest(p=p), self.assertRaises(ValueError):
                thin_counts(x, p, 42)
        for seed in (-1, 2.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                thin_counts(x, .1, seed)
        for scale in (0, -.1, np.nan, np.inf, True):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                warp_counts(x, scale)
        with self.assertRaises(ValueError):
            thin_counts([[.1]], .1, 42)
        with self.assertRaises(ValueError):
            warp_counts([[.1]], 1.1)


if __name__ == '__main__':
    unittest.main()
