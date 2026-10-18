import copy
import json
from pathlib import Path
import tempfile
import unittest

from cost_records import seed_directory, record_path, load_measurement, normalized_record


class CostRecordTests(unittest.TestCase):
    def test_canonical_locations_match_training_layout(self):
        root = Path('checkout')
        self.assertEqual(record_path(root, 'DA_SE_adLIF', 'SHD', 42),
                         root / 'results/valid/DA_SE_adLIF/SHD/seed_42/COST_RESULTS.json')
        self.assertEqual(record_path(root, 'SE_adLIF', 'ECG', 123),
                         root / 'results/valid/reference/ECG/seed_123/COST_RESULTS.json')
        self.assertEqual(seed_directory(root, 'MR_SE_adLIF', 'SHD', 456),
                         root / 'results/valid/MR_SE_adLIF/K_2/SHD/seed_456')

    def test_explicit_separate_benchmark_layout_is_preserved(self):
        self.assertEqual(record_path('.', 'DA_SE_adLIF', 'SHD', 42, Path('custom')),
                         Path('custom/DA_SE_adLIF/seed_42.json'))
        self.assertEqual(record_path('.', 'SE_adLIF', 'SSC', 123, Path('custom')),
                         Path('custom/SSC/seed_123.json'))

    def test_legacy_totals_are_derived_without_mutating_record(self):
        record = dict(version='shd_costs_v1', seed=42, measurements=dict(
            reproduced_test_acc=0.9, totals=dict(samples=2),
            dense_gflops_test_padded=1.0, nonmatrix_ops_per_sample=20,
            nonmatrix_scalar_operations_by_kind=dict(add=12, exp=28)))
        original = copy.deepcopy(record)
        result = normalized_record(record)
        self.assertEqual(record, original)
        self.assertEqual(result['version'], 'shd_costs_v1')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['dataset'], 'SHD')
        self.assertAlmostEqual(result['measurements']['arithmetic_gflops_test_padded'], 1 + 6e-9)
        self.assertAlmostEqual(result['measurements']['logical_gops_test_padded'], 1 + 20e-9)

    def test_read_and_normalize_keeps_original_file_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = record_path(tmp, 'SE_adLIF', 'SSC', 42)
            path.parent.mkdir(parents=True)
            record = dict(version='reference_test_costs_v1', dataset='SSC', seed=42,
                          measurements=dict(reproduced_test_acc=0.8, arithmetic_gflops_test_padded=1.1,
                                            logical_gops_test_padded=1.2))
            path.write_text(json.dumps(record, indent=2), encoding='utf-8')
            before = path.read_bytes()
            normalized_record(load_measurement(tmp, 'SE_adLIF', 'SSC', 42))
            self.assertEqual(before, path.read_bytes())

    def test_unknown_schemas_and_unverified_legacy_measurements_fail(self):
        for record in [dict(version='unknown'), dict(version='shd_costs_v1', measurements={})]:
            with self.assertRaises(ValueError):
                normalized_record(record)


if __name__ == '__main__':
    unittest.main()
