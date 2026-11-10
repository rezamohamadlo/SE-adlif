"""Non-training tests for read-only SHD validation framing and selection."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
import torch
from tonic.datasets.hsd import SHD
from tonic.io import make_structured_array
from tonic.transforms import Compose, Downsample, ToFrame

from datasets.utils.pad_tensors import PadTensors
from scripts.adaptation_analysis.shd_data import (
    SHDValidationData, preflight_shd, select_validation_ids,
    validate_shd_config, validation_indices,
)


class SHDDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='shd-validation-fixture-')
        self.root = Path(self.temp.name)
        self.path = self.root/'SHDWrapper/shd_train.h5'
        self.path.parent.mkdir()
        with h5py.File(self.path, 'w') as dataset:
            dataset['labels'] = np.arange(200, dtype=np.int64) % 4
            dataset['extra/speaker'] = np.arange(200, dtype=np.int64) % 10
            times = dataset.create_dataset('spikes/times', shape=(200,), dtype=h5py.vlen_dtype(np.dtype('float16')))
            units = dataset.create_dataset('spikes/units', shape=(200,), dtype=h5py.vlen_dtype(np.dtype('uint16')))
            for index in range(200):
                times[index] = np.array([0, .001, .002, .004, .008, .012, .016], dtype=np.float16)
                units[index] = np.array([0, 4, 5, 699, 30, 40, 50], dtype=np.uint16)
        self.sha = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.cfg = dict(dataset=dict(_target_='datasets.shd.SHDLDM',data_path=str(self.root),
                                    validate_on=.2,random_seed=42,spatial_factor=.2,time_factor=.001,
                                    window_size=4,pad_to_min_size=250,ignore_first_timesteps=10,num_classes=20))

    def tearDown(self):
        assert self.root.resolve().parent == Path(tempfile.gettempdir()).resolve()
        self.temp.cleanup()

    def test_validation_order_exactly_matches_torch_random_split(self):
        _, val = torch.utils.data.random_split(range(200), [160, 40], generator=torch.Generator().manual_seed(42))
        self.assertEqual(validation_indices(200), val.indices)
        self.assertEqual(len(val), 40)

    def test_subset_deterministic_balanced_and_only_validation(self):
        first = select_validation_ids(self.path, samples=12)
        self.assertEqual(first, select_validation_ids(self.path, samples=12))
        self.assertEqual(len(set(first)), 12)
        val = validation_indices(200)
        self.assertEqual(first, [i for i in val if i in first])
        with h5py.File(self.path, 'r') as dataset:
            labels = np.array([dataset['labels'][i] for i in first])
        self.assertEqual(np.bincount(labels).tolist(), [3,3,3,3])
        with self.assertRaises(ValueError):
            select_validation_ids(self.path, samples=41)

    def test_preflight_checks_hash_without_creating_cache_or_test_data(self):
        before = sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*'))
        data = preflight_shd(self.cfg, self.sha)
        self.assertEqual(data['validation_samples'], 40)
        self.assertEqual(data['train_samples'], 160)
        self.assertFalse(data['official_test_opened'])
        self.assertEqual(data['sha256'], self.sha)
        with self.assertRaises(ValueError):
            preflight_shd(self.cfg, '0'*64)
        self.assertEqual(before, sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob('*')))

    def test_repository_relative_data_path_resolves_after_working_directory_change(self):
        from project_paths import ROOT, portable_path
        relative = portable_path(ROOT / 'data')
        original = Path.cwd()
        try:
            os.chdir(self.root)
            from project_paths import local_path
            self.assertEqual(local_path(relative), ROOT / 'data')
            data = SHDValidationData(self.cfg, samples=1, expected_sha256=self.sha)
            self.assertEqual(len(data), 1)
        finally:
            os.chdir(original)

    def test_config_rejects_other_split_or_augmentation(self):
        self.assertEqual(validate_shd_config(self.cfg)['random_seed'], 42)
        for key,value in [('validate_on','speaker_holdout'),('spatial_factor',.1),('random_seed',123)]:
            changed = dict(self.cfg['dataset']);changed[key]=value
            with self.assertRaises(ValueError):
                validate_shd_config(changed)
        changed=dict(self.cfg['dataset'], train_augmentation={'version':'something'})
        with self.assertRaises(ValueError):validate_shd_config(changed)

    def test_frames_equal_saved_tonic_pipeline_and_padding_separate(self):
        data = SHDValidationData(self.cfg, samples=3, expected_sha256=self.sha)
        sample_id = data.sample_ids[0]
        sample = data.load(sample_id)
        with h5py.File(self.path,'r') as h:
            events=make_structured_array(h['spikes/times'][sample_id].astype(np.float64)*1e6,
                                         h['spikes/units'][sample_id],1,dtype=SHD.dtype)
        expected=Compose([Downsample(time_factor=.001,spatial_factor=.2),ToFrame(sensor_size=(140,1,1),time_window=4)])(events).reshape(-1,140)
        np.testing.assert_array_equal(sample['counts'],expected)
        self.assertEqual(sample['support_frames'],len(expected))
        self.assertEqual(sample['stored_frames'],250)
        self.assertEqual(sample['raw_event_count'],7)
        x,y,blocks=data.model_item(sample_id)
        np.testing.assert_array_equal(x[:len(expected)].numpy(),expected)
        self.assertTrue(torch.equal(x[len(expected):],torch.zeros_like(x[len(expected):])))
        self.assertEqual(x.shape,(250,140))
        self.assertEqual(y.dtype,torch.int64)
        self.assertEqual(blocks.dtype,torch.int64)
        self.assertTrue(torch.equal(blocks[:10],torch.zeros(10,dtype=torch.int64)))
        self.assertTrue(torch.equal(blocks[10:],torch.ones(240,dtype=torch.int64)))

    def test_collation_keeps_dummy_target_and_batch_padding_mask(self):
        data=SHDValidationData(self.cfg,samples=2,expected_sha256=self.sha)
        first=data.model_item(data.sample_ids[0])
        second=data.model_item(data.sample_ids[1],np.zeros((270,140),dtype=np.int64))
        inputs,targets,blocks=PadTensors()([first,second])
        self.assertEqual(inputs.shape,(2,270,140))
        self.assertEqual(targets.shape,(2,2))
        self.assertTrue(torch.equal(targets[:,0],torch.full((2,),-1,dtype=torch.int64)))
        self.assertTrue(torch.equal(blocks[0,250:],torch.zeros(20,dtype=torch.int64)))
        self.assertTrue(torch.equal(blocks[1,250:],torch.ones(20,dtype=torch.int64)))

    def test_explicit_ids_must_match_validation_and_are_not_silently_substituted(self):
        val=validation_indices(200)
        data=SHDValidationData(self.cfg,sample_ids=val[:2],expected_sha256=self.sha)
        self.assertEqual(data.sample_ids,val[:2])
        training_id=next(i for i in range(200) if i not in val)
        with self.assertRaises(ValueError):
            SHDValidationData(self.cfg,sample_ids=[training_id],expected_sha256=self.sha)
        with self.assertRaises(ValueError):data.load(training_id)
        with self.assertRaises(ValueError):data.model_item(val[0],np.full((4,140),.5))


if __name__=='__main__':
    unittest.main()
