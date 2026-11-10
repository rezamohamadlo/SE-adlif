"""Pure pairing/manifest/path checks; never fit, infer, or iterate datasets."""
import copy
import os
from pathlib import Path
import tempfile
import unittest

from project_paths import ROOT, local_path
from scripts.adaptation_analysis.pilot import (
    output_directory,
    paired_rows,
    validate_manifest,
    cost_model_identity,
)


OUTPUT_ROOT = 'results/experiments/adaptation_mechanism_analysis_v1'


def record(sample_id='record-1', *, target=2, loss=.75, correct=False,
           input_sha256='a' * 64, condition='clean', replicate=0):
    return dict(sample_id=sample_id, target=target, loss=loss, correct=correct,
                input_sha256=input_sha256, condition=condition, replicate=replicate)


class PairedRowIntegrityTests(unittest.TestCase):
    def test_pairing_has_correct_loss_and_accuracy_signs(self):
        parent = [record('a', loss=.75, correct=False),
                  record('b', loss=.25, correct=True),
                  record('c', loss=.5, correct=True)]
        variant = [record('a', loss=.5, correct=True),
                   record('b', loss=.75, correct=False),
                   record('c', loss=.5, correct=True)]
        original = copy.deepcopy((parent, variant))
        by_id = {r['sample_id']: r for r in paired_rows(parent, variant)}
        self.assertEqual(set(by_id), {'a', 'b', 'c'})
        self.assertAlmostEqual(by_id['a']['benefit'], .25)
        self.assertEqual(by_id['a']['correctness_difference'], 1)
        self.assertAlmostEqual(by_id['b']['benefit'], -.5)
        self.assertEqual(by_id['b']['correctness_difference'], -1)
        self.assertEqual(by_id['c']['benefit'], 0)
        self.assertEqual(by_id['c']['correctness_difference'], 0)
        self.assertEqual((parent, variant), original)

    def test_pairs_match_samples_even_if_input_order_differs(self):
        parent = [record('a', target=1, loss=.9), record('b', target=3, loss=.4)]
        variant = [record('b', target=3, loss=.3), record('a', target=1, loss=.7)]
        rows = {r['sample_id']: r for r in paired_rows(parent, variant)}
        self.assertAlmostEqual(rows['a']['benefit'], .2)
        self.assertAlmostEqual(rows['b']['benefit'], .1)

    def test_duplicate_samples_in_either_model_are_rejected(self):
        one = record()
        for parent, variant in [([one, copy.deepcopy(one)], [one]),
                                ([one], [one, copy.deepcopy(one)]),
                                ([one, copy.deepcopy(one)], [one, copy.deepcopy(one)])]:
            with self.subTest(parent_n=len(parent), variant_n=len(variant)):
                with self.assertRaises(ValueError):
                    paired_rows(parent, variant)

    def test_missing_or_different_samples_are_rejected(self):
        for parent, variant in [([record('a')], []), ([], [record('a')]),
                                ([record('a')], [record('b')]),
                                ([record('a'), record('b')], [record('a')])]:
            with self.subTest(parent=parent, variant=variant):
                with self.assertRaises(ValueError):
                    paired_rows(parent, variant)

    def test_target_input_and_perturbation_identity_must_match(self):
        differences = {'target': 3, 'input_sha256': 'b' * 64,
                       'condition': 'thin_0.1', 'replicate': 1}
        for field, changed in differences.items():
            parent, variant = record(), record(loss=.25)
            variant[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                paired_rows([parent], [variant])

    def test_explicit_training_seeds_must_match(self):
        parent, variant = dict(record(), seed=42), dict(record(), seed=123)
        with self.assertRaises(ValueError):
            paired_rows([parent], [variant])


class ManifestIntegrityTests(unittest.TestCase):
    def test_current_and_historical_cost_model_identity(self):
        self.assertEqual(cost_model_identity({'model': 'MT_SE_adLIF'}), 'MT_SE_adLIF')
        self.assertEqual(cost_model_identity({'saved_config': {'model_variant': 'AMT_SE_adLIF'}}), 'AMT_SE_adLIF')
        with self.assertRaises(ValueError):
            cost_model_identity({'saved_config': {}})

    def setUp(self):
        self.manifest = dict(protocol='adaptation_mechanism_analysis_v1',
                             dataset='SHD', split_seed=42,
                             training_seeds=[42, 123, 456], tta=False,
                             features={'short_ms':20, 'long_ms':200},
                             checkpoint={'path':'results/example.ckpt',
                                         'sha256':'a' * 64})

    def test_equal_manifest_is_accepted_without_mutation(self):
        existing = copy.deepcopy(self.manifest)
        requested = copy.deepcopy(self.manifest)
        validate_manifest(existing, requested)
        self.assertEqual(existing, self.manifest)
        self.assertEqual(requested, self.manifest)

    def test_any_effective_manifest_difference_is_rejected(self):
        cases = []
        changed = copy.deepcopy(self.manifest); changed['features']['short_ms'] = 40
        cases.append(changed)
        changed = copy.deepcopy(self.manifest); changed['checkpoint']['sha256'] = 'b' * 64
        cases.append(changed)
        changed = copy.deepcopy(self.manifest); changed['training_seeds'].reverse()
        cases.append(changed)
        changed = copy.deepcopy(self.manifest); changed['tta'] = True
        cases.append(changed)
        changed = copy.deepcopy(self.manifest); changed['new_setting'] = 1
        cases.append(changed)
        changed = copy.deepcopy(self.manifest); del changed['split_seed']
        cases.append(changed)
        for index, requested in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_manifest(copy.deepcopy(self.manifest), requested)


class DiagnosticOutputPathTests(unittest.TestCase):
    def test_each_pc_receives_distinct_project_relative_output(self):
        paths = [Path(output_directory(OUTPUT_ROOT, pc)).resolve()
                 for pc in ('pc1', 'pc2', 'pc3')]
        self.assertEqual(len(set(paths)), 3)
        for pc, path in zip(('pc1', 'pc2', 'pc3'), paths):
            self.assertEqual(path, (local_path(OUTPUT_ROOT)/pc).resolve())
            self.assertTrue(path.is_relative_to((ROOT/'results/experiments').resolve()))

    def test_absolute_and_registered_old_checkout_paths_are_portable(self):
        expected = (local_path(OUTPUT_ROOT)/'pc2').resolve()
        for root in [ROOT/OUTPUT_ROOT, 'E:/AI_paper_nazari/SE-adlif/' + OUTPUT_ROOT]:
            with self.subTest(root=str(root)):
                self.assertEqual(Path(output_directory(root, 'pc2')).resolve(), expected)

    def test_hydra_changed_working_directory_does_not_redirect_output(self):
        previous = Path.cwd()
        expected = (local_path(OUTPUT_ROOT)/'pc1').resolve()
        with tempfile.TemporaryDirectory(prefix='adaptation-path-') as folder:
            try:
                os.chdir(folder)
                self.assertEqual(Path(output_directory(OUTPUT_ROOT, 'pc1')).resolve(), expected)
            finally:
                os.chdir(previous)

    def test_external_valid_roots_aliases_and_traversal_are_rejected(self):
        paths = ['results/valid/adaptation_diagnostic',
                 'results/experiments/../../outside_results',
                 'results/experiments/../valid/overwrite',
                 'results/experiments/amt_v1',
                 'results/experiments/ssc_loss_diagnostic_v1/mean_membrane_potentials/seed_42',
                 ROOT.parent/'outside_diagnostics']
        for path in paths:
            with self.subTest(path=str(path)), self.assertRaises(ValueError):
                output_directory(path, 'pc1')

    def test_pc_name_cannot_escape_or_alias_output(self):
        for pc in ('', 'pc4', '../pc1', 'pc1/other', '.', '..'):
            with self.subTest(pc=pc), self.assertRaises(ValueError):
                output_directory(OUTPUT_ROOT, pc)


if __name__ == '__main__':
    unittest.main()
