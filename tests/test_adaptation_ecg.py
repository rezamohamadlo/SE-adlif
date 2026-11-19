"""User-launched contracts; no training, dataset downloads or checkpoint forward."""
import json
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from project_paths import local_path, portable_path
from scripts.adaptation_ecg.data import ValidationData, array_hash, delete_events, load_validation
from scripts.adaptation_ecg.preflight import guard
from scripts.adaptation_ecg.runtime import aggregate, check_anchor, score_outputs


class ECGContracts(unittest.TestCase):
    def test_signed_deletion_preserves_polarity_and_alignment(self):
        x = np.tile(np.array([[-1, 0, 1, 1]], dtype=np.float32), (100, 1))
        z = delete_events(x, .5, 17, 0, 234)
        self.assertEqual(z.shape, x.shape)
        self.assertTrue(np.all((z == 0) | (z == x)))
        self.assertTrue(np.any(z == -1) and np.any(z == 1) and np.any(z != x))
        self.assertTrue(np.array_equal(delete_events(x, 0, 17, 0, 234), x))
        self.assertFalse(delete_events(x, 1, 17, 0, 234).any())

    def test_deletion_uses_sample_and_replicate_not_model_or_seed(self):
        x = np.ones((200, 4), dtype=np.float32)
        a = delete_events(x, .1, 17, 0, 234)
        self.assertTrue(np.array_equal(a, delete_events(x, .1, 17, 0, 234)))
        self.assertFalse(np.array_equal(a, delete_events(x, .1, 18, 0, 234)))
        self.assertFalse(np.array_equal(a, delete_events(x, .1, 17, 1, 234)))

    def test_temporal_targets_and_blocks_unchanged_by_deletion(self):
        x = np.ones((2, 6, 4), dtype=np.float32)
        y = np.array([[0, 1, 2, 3, 4, 5]]*2, dtype=np.int64)
        data = ValidationData(x, y, [1])
        spec = dict(deletion_probability=.1, selection_seed=3)
        a = data.batch([1], 'clean', 0, spec)
        b = data.batch([1], 'delete_0.1', 2, spec)
        self.assertTrue(torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]))
        self.assertEqual(a[1].tolist(), [[-1, 0, 1, 2, 3, 4, 5]])
        self.assertEqual(a[2].tolist(), [[1, 2, 3, 4, 5, 6]])
        with self.assertRaises(ValueError): data.input(0, 'clean', 0, spec)

    def test_temporal_ce_and_accuracy_ignore_dummy_and_padding(self):
        outputs = torch.tensor([[[2., 0.], [0., 2.], [999., -999.]],
                                [[0., 2.], [0., 2.], [2., 0.]]])
        targets = torch.tensor([[-1, 0, 1, -1], [-1, 1, 0, 0]], dtype=torch.long)
        blocks = torch.tensor([[1, 2, 0], [1, 2, 3]])
        scores = score_outputs(outputs, targets, blocks)
        expected = torch.nn.functional.cross_entropy(outputs[0, :2], targets[0, 1:3])
        self.assertAlmostEqual(scores[0]['loss'], float(expected), places=6)
        self.assertEqual([r['correct_steps'] for r in scores], [2, 2])
        self.assertEqual([r['valid_steps'] for r in scores], [2, 3])

    def test_mask_does_not_silently_accept_unmapped_targets(self):
        with self.assertRaises(ValueError):
            score_outputs(torch.zeros((1, 2, 2)), torch.tensor([[-1, 0, 1]]), torch.tensor([[1, 1]]))

    def test_record_and_step_weighting_are_distinct(self):
        rows = [dict(valid_steps=1, loss_sum=1., loss=1., correct_steps=1, accuracy=1.),
                dict(valid_steps=3, loss_sum=9., loss=3., correct_steps=0, accuracy=0.)]
        result = aggregate(rows)
        self.assertEqual(result['step_weighted_loss'], 2.5)
        self.assertEqual(result['record_mean_loss'], 2.)
        self.assertEqual(result['step_weighted_accuracy'], .25)
        self.assertEqual(result['record_mean_accuracy'], .5)

    def test_failed_clean_anchor_blocks_diagnostics(self):
        rows = [dict(valid_steps=1, loss_sum=1., loss=1., correct_steps=1, accuracy=1.)]
        run = dict(model='MT_SE_adLIF', seed=42, selected_epoch=2, validation=dict(accuracy=0., loss=1.))
        with self.assertRaises(ValueError):
            check_anchor(rows, run, dict(accuracy_tolerance=1e-6, loss_tolerance=.001))

    def test_manifest_guard_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'MANIFEST.json'
            path.write_text(json.dumps(dict(version=1)))
            before = path.read_bytes()
            with self.assertRaises(ValueError): guard(root, dict(version=2))
            self.assertEqual(path.read_bytes(), before)

    def test_portable_paths_do_not_depend_on_working_directory(self):
        path = local_path('config/dataset/ecg.yaml', must_exist=True)
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.assertEqual(portable_path(path), 'config/dataset/ecg.yaml')
                self.assertEqual(local_path(portable_path(path)), path)
                self.assertEqual(local_path('data/QTDB/QTDB_train.mat', root=directory),
                                 Path(directory).resolve()/'data/QTDB/QTDB_train.mat')
            finally:
                os.chdir(previous)

    def test_train_only_loader_preserves_zero_vector_argmax(self):
        x = np.zeros((40, 5, 4), dtype=np.int16)
        y = np.zeros((40, 5, 6), dtype=np.uint8)
        y[:, 1:, 3] = 1
        spec = dict(dataset_file='data/QTDB/QTDB_train.mat', dataset_sha256='fake',
                    expected_input_shape=[40, 5, 4], expected_validation_samples=2)
        baseline = dict(datasets=dict(ECG=dict(valid_fraction=.05)), reproduction_policy=dict(split_seed=42))
        fake = dict(x=x, y=y, t=np.broadcast_to(np.arange(5), (40, 5)))
        with patch('scripts.adaptation_ecg.data.local_path', return_value=Path('QTDB_train.mat')), \
                patch('scripts.adaptation_ecg.data.loadmat', return_value=fake) as reader, \
                patch('scripts.adaptation_analysis.pilot.sha256', return_value='fake'):
            data, metadata = load_validation(spec, baseline, dict(dataset={}))
        self.assertEqual(reader.call_count, 1)
        self.assertEqual(Path(reader.call_args.args[0]).name, 'QTDB_train.mat')
        self.assertEqual(data.y[data.ids[0]].tolist(), [0, 3, 3, 3, 3])
        self.assertEqual(metadata['zero_label_positions_validation'], 2)

    def test_array_identity_includes_shape_and_dtype(self):
        x = np.arange(6, dtype=np.int64)
        self.assertNotEqual(array_hash(x), array_hash(x.reshape(2, 3)))
        self.assertNotEqual(array_hash(x), array_hash(x.astype(np.float64)))


if __name__ == '__main__':
    unittest.main()
