"""User-launched SSC loader, streaming and scope contracts; no training."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

from scripts.adaptation_ssc import evaluate
from scripts.adaptation_ssc.data import SSCValidationData, check_legacy_contract, select_ids, validate_config
from scripts.adaptation_ssc.inputs import InputSequence, input_rows
from scripts.adaptation_ssc.runtime import check_rows


class SSCPilotContracts(unittest.TestCase):
    def setUp(self):
        self.study, self.baseline = evaluate.load_study()
        self.cfg = dict(dataset=dict(_target_='datasets.ssc.SSCLDM', num_classes=35,
            data_path='data', **{k: self.baseline['datasets']['SSC'][k] for k in (
                'spatial_factor', 'time_factor', 'window_size', 'pad_to_min_size',
                'ignore_first_timesteps', 'batch_size')}))

    @contextlib.contextmanager
    def synthetic_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ssc_valid.h5'
            with h5py.File(path, 'w') as f:
                times = f.create_dataset('spikes/times', (3,), dtype=h5py.vlen_dtype(np.dtype('float16')))
                units = f.create_dataset('spikes/units', (3,), dtype=h5py.vlen_dtype(np.dtype('uint16')))
                for sid in range(3):
                    times[sid] = np.linspace(.1, .9, 30+sid).astype(np.float16)
                    units[sid] = (np.arange(30+sid)*23 % 700).astype(np.uint16)
                f.create_dataset('labels', data=np.array([0, 1, 2], dtype=np.int64))
            meta = dict(path=path.as_posix(), validation_indices=[0, 1, 2], min_frames=300,
                        ignore_first_timesteps=10, bin_ms=4)
            yield SSCValidationData(self.cfg, meta, [0, 1, 2], ['speaker-a']*3, [0, 1, 2])

    def test_timestamp_widening_matches_wrapper_without_dataset_setup(self):
        with self.synthetic_data() as data, patch('tonic.datasets.SSC.__init__', side_effect=AssertionError('constructor')), \
                patch('datasets.ssc.SSCLDM.setup', side_effect=AssertionError('setup')):
            result = check_legacy_contract(data)
            self.assertTrue(result['matches_saved_wrapper'])
            x, target, blocks = data.model_item(0)
            self.assertEqual(tuple(x.shape), (300, 140))
            self.assertTrue(torch.isfinite(x).all())
            self.assertGreater(x.sum().item(), 0)
            self.assertEqual(target.dtype, torch.long)
            self.assertTrue(torch.equal(blocks[:10], torch.zeros(10, dtype=torch.long)))
            self.assertTrue(torch.equal(blocks[10:], torch.ones(290, dtype=torch.long)))

    def test_ssc_rejects_shd_holdout_and_padding(self):
        validate_config(self.cfg, self.baseline)
        for changes in ({'validate_on': .2}, {'pad_to_min_size': 250}, {'additional_test_set_validation': True}):
            cfg = copy.deepcopy(self.cfg)
            cfg['dataset'].update(changes)
            with self.assertRaises(ValueError):
                validate_config(cfg, self.baseline)

    def test_pilot_support_excludes_minimum_padding(self):
        with self.synthetic_data() as data:
            sid, counts, row = input_rows(data, [0], ('clean', 0, 'identity', 1.), self.study)[0]
            self.assertLess(len(counts), 300)
            self.assertEqual(row['stored_frames'], 300)
            self.assertEqual(row['duration_ms'], len(counts)*4)
            self.assertEqual(row['original_support_frames'], len(counts))
            self.assertEqual(len(data.model_item(sid, counts)[0]), 300)

    def test_streaming_reconstructs_only_requested_batch_in_order(self):
        with self.synthetic_data() as data, patch.object(data, 'load', wraps=data.load) as loader:
            sequence = InputSequence(data, [2, 0, 1], ('clean', 0, 'identity', 1.), self.study)
            batch = sequence[:2]
            self.assertEqual([r[0] for r in batch], [2, 0])
            self.assertEqual([c.args[0] for c in loader.call_args_list], [2, 0])

    def test_streaming_rejects_changed_cached_input(self):
        with self.synthetic_data() as data:
            condition = ('clean', 0, 'identity', 1.)
            row = input_rows(data, [0], condition, self.study)[0][2]
            row['input_sha256'] = 'changed'
            with self.assertRaises(ValueError):
                InputSequence(data, [0], condition, self.study, {0: row})[:1]

    def test_class_balancing_and_original_order_are_fixed(self):
        labels = np.repeat(np.arange(35), 10)
        meta = dict(validation_indices=list(range(350)))
        ids = select_ids(labels, meta, 256, self.study['selection_seed'])
        self.assertEqual(ids, sorted(set(ids)))
        self.assertEqual(ids, select_ids(labels, meta, 256, self.study['selection_seed']))
        counts = np.bincount(labels[ids], minlength=35)
        self.assertEqual(len(ids), 256)
        self.assertLessEqual(int(counts.max()-counts.min()), 1)

    def test_prediction_cache_requires_order_and_model_identity(self):
        run = dict(model='MT_SE_adLIF', seed=42)
        rows = [dict(sample_id=sid, condition='clean', replicate=0, **run) for sid in [2, 3]]
        check_rows(rows, [2, 3], run, ('clean', 0, 'identity', 1.))
        with self.assertRaises(ValueError):
            check_rows(rows, [3, 2], run, ('clean', 0, 'identity', 1.))
        rows[0]['seed'] = 123
        with self.assertRaises(ValueError):
            check_rows(rows, [2, 3], run, ('clean', 0, 'identity', 1.))

    def test_check_only_never_enters_runtime(self):
        with patch.object(evaluate, 'prepare', return_value=({}, None, {}, None)), \
                patch('scripts.adaptation_ssc.runtime.execute', side_effect=AssertionError('execution')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(['--check-only']), 0)


if __name__ == '__main__':
    unittest.main()
