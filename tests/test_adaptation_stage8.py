"""USER-RUN ONLY report contracts; no neural inference or training."""
import unittest

from scripts.adaptation_stage8.evidence import clean_groups, test_accuracy as saved_test_accuracy
from scripts.adaptation_stage8.reporting import aggregate, feature_values, paired


class SynthesisContracts(unittest.TestCase):
    def test_sample_sd_is_across_distinct_training_seeds(self):
        result = aggregate([dict(seed=s, value=v) for s, v in zip([42, 123, 456], [1, 2, 3])], 'value')
        self.assertEqual(result, dict(mean=2, sample_sd=1, n_training_seeds=3))

    def test_repeated_seed_cannot_replace_missing_seed(self):
        with self.assertRaisesRegex(ValueError, 'three distinct'):
            aggregate([dict(seed=s, value=1) for s in [42, 42, 123]], 'value')

    def test_nonfinite_seed_value_fails(self):
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            aggregate([dict(seed=s, value=float('nan')) for s in [42, 123, 456]], 'value')

    def test_acf_exclusions_do_not_impute_zero(self):
        values, counts = feature_values([
            dict(autocorr_tau_ms=0, autocorr_censored=False),
            dict(autocorr_tau_ms=200, autocorr_censored=True),
            dict(autocorr_tau_ms=None, autocorr_reason='constant'),
            dict(autocorr_tau_ms=None)], 'autocorr_tau_ms')
        self.assertEqual(values, [0])
        self.assertEqual((counts['finite_uncensored'], counts['censored'], counts['invalid'], counts['missing_or_nonfinite']), (1, 1, 1, 1))

    def test_fast_slow_invalid_flag_is_preserved(self):
        values, counts = feature_values([dict(fast_slow_contrast=5, fast_slow_valid=False)], 'fast_slow_contrast')
        self.assertEqual(values, [])
        self.assertEqual(counts['invalid'], 1)

    def test_single_test_result_matches_saved_reproduction(self):
        self.assertEqual(saved_test_accuracy([dict(test_acc='0.9'), dict(test_acc='')], .90000002), .9)

    def test_mixed_test_history_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            saved_test_accuracy([dict(test_acc='.9'), dict(test_acc='.8')], .9)

    def test_cost_accuracy_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'disagree'):
            saved_test_accuracy([dict(test_acc='.9')], .91)

    def test_clean_groups_require_complete_seed_cohort(self):
        with self.assertRaisesRegex(ValueError, 'Clean seed'):
            clean_groups(dict(groups=[]))

    def test_paired_sd_uses_seed_differences(self):
        rows = [dict(dataset='SHD', model=model, seed=seed, measurements=dict(cost=value))
                for model, values in [('parent', [1, 3, 5]), ('variant', [3, 5, 7])]
                for seed, value in zip([42, 123, 456], values)]
        self.assertEqual(paired(rows, 'SHD', 'parent', 'variant', 'cost'),
                         dict(mean=2, sample_sd=0, n_training_seeds=3))


if __name__ == '__main__':
    unittest.main()
