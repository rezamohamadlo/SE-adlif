import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_setup', ROOT / 'scripts/verify_setup.py')
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class SetupTests(unittest.TestCase):
    def test_repository_pins_include_chosen_numpy_and_cuda(self):
        pins = setup.package_pins(ROOT / 'requirements.txt')
        self.assertEqual(pins['numpy'], '1.26.4')
        self.assertEqual(pins['torch'], '2.6.0+cu124')
        self.assertEqual(pins['pytorch-lightning'], '2.2.1')

    def test_version_drift_and_missing_dependencies_are_errors(self):
        result = setup.mismatches({'numpy': '1.26.4', 'torch': '2.6.0+cu124'}, {'numpy': '2.4.6'})
        self.assertEqual(result['numpy']['installed'], '2.4.6')
        self.assertIsNone(result['torch']['installed'])
        self.assertEqual(setup.mismatches({'numpy': '1.26.4'}, {'numpy': '1.26.4'}), {})

    def test_unpinned_or_duplicate_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'requirements.txt'
            for text in ['numpy>=1.26', 'numpy==1.26.4\nnumpy==2.4.6']:
                p.write_text(text)
                with self.assertRaises(ValueError):
                    setup.package_pins(p)

    def test_data_inventory_does_not_download_or_create_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inventory = setup.data_inventory(root)
            self.assertFalse(any(v for group in inventory.values() for v in group.values()))
            self.assertEqual(list(root.iterdir()), [])

    def test_config_fingerprints_survive_git_line_ending_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a.yaml', Path(tmp) / 'b.yaml'
            a.write_bytes(b'batch: 256\nseed: 42\n')
            b.write_bytes(b'batch: 256\r\nseed: 42\r\n')
            self.assertEqual(setup.text_hash(a), setup.text_hash(b))

    def test_verifier_rejects_missing_packages_without_importing_cuda_or_writing(self):
        with patch.object(setup.metadata, 'version', side_effect=setup.metadata.PackageNotFoundError), \
             patch.object(setup.subprocess, 'check_output', side_effect=['same-commit\n', '']):
            report = setup.verify(expected_commit='same-commit')
        self.assertFalse(report['environment_ready'])
        self.assertTrue(report['package_mismatches'])
        self.assertNotIn('gpu', report)


if __name__ == '__main__':
    unittest.main()
