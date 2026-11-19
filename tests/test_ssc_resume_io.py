"""User-launched atomic I/O contracts; no scientific experiment execution."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from scripts.adaptation_analysis import pilot
from scripts.ssc_resume_io import AtomicJSONWriter, filesystem_preflight


def locked(code=32):
    error = PermissionError('simulated Windows file lock')
    error.winerror = code
    return error


class AtomicJSONContracts(unittest.TestCase):
    def test_original_json_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = dict(text='unicode \u06a9', values=[None, True, 1.25], nested=dict(x=2))
            pilot.write_json(root / 'original.json', data)
            writer = AtomicJSONWriter(root, pilot.write_json)
            writer(root / 'compatible.json', data)
            self.assertEqual((root / 'original.json').read_bytes(), (root / 'compatible.json').read_bytes())

    def test_transient_lock_retries_without_changing_old_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'PROGRESS.json'
            target.write_text('old complete payload')
            calls, sleeps = [], []
            def replace(source, destination):
                calls.append(source)
                self.assertEqual(target.read_text(), 'old complete payload')
                if len(calls) == 1:
                    raise locked(5)
                os.replace(source, destination)
            writer = AtomicJSONWriter(root, pilot.write_json, replace=replace, sleep=sleeps.append)
            writer(target, dict(status='new'))
            self.assertEqual(json.loads(target.read_text()), dict(status='new'))
            self.assertEqual(len(calls), 2)
            self.assertEqual(len(sleeps), 1)
            self.assertEqual(writer.retries[0]['winerror'], 5)
            self.assertFalse(writer.last_temporary.exists())

    def test_persistent_lock_is_bounded_and_retains_both_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'PROGRESS.json'
            target.write_bytes(b'old')
            replace = Mock(side_effect=locked())
            sleep = Mock()
            writer = AtomicJSONWriter(root, pilot.write_json, replace=replace, sleep=sleep, attempts=3)
            with self.assertRaises(PermissionError):
                writer(target, dict(status='uncommitted'))
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(target.read_bytes(), b'old')
            self.assertEqual(json.loads(writer.last_temporary.read_text()), dict(status='uncommitted'))

    def test_non_windows_permission_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replace = Mock(side_effect=PermissionError('persistent permission problem'))
            sleep = Mock()
            writer = AtomicJSONWriter(root, pilot.write_json, replace=replace, sleep=sleep)
            with self.assertRaises(PermissionError):
                writer(root / 'data.json', dict(x=1))
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()

    def test_other_roots_and_non_json_writes_are_delegated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'suite'
            root.mkdir()
            fallback = Mock()
            writer = AtomicJSONWriter(root, fallback)
            writer(Path(directory) / 'outside.json', dict(x=1))
            writer(root / 'outside_scope.txt', dict(x=1))
            self.assertEqual(fallback.call_count, 2)
            self.assertEqual(writer.writes, 0)

    def test_fixed_temporary_from_prior_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stale = root / 'PROGRESS.json.tmp'
            stale.write_bytes(b'previous failure evidence')
            writer = AtomicJSONWriter(root, pilot.write_json)
            writer(root / 'PROGRESS.json', dict(status='new'))
            self.assertEqual(stale.read_bytes(), b'previous failure evidence')
            self.assertNotEqual(writer.last_temporary, stale)

    def test_nonfinite_data_is_rejected_before_creating_a_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AtomicJSONWriter(root, pilot.write_json)
            with self.assertRaises(ValueError):
                writer(root / 'invalid.json', dict(value=float('nan')))
            self.assertEqual(list(root.iterdir()), [])

    def test_probe_does_not_touch_existing_scientific_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'MANIFEST.json'
            manifest.write_bytes(b'original frozen manifest')
            record = filesystem_preflight(root, pilot.write_json)
            self.assertEqual(record['status'], 'passed')
            self.assertFalse(record['scientific_files_touched'])
            self.assertEqual(manifest.read_bytes(), b'original frozen manifest')


if __name__ == '__main__':
    unittest.main()
