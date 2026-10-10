import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from datasets.shd import SHDLDM


class FakeSHD:
    sensor_size = (700, 1, 1)

    def __init__(self, **kwargs):
        self.train = kwargs['train']

    def __len__(self):
        return 100


class SHDSplitCacheTests(unittest.TestCase):
    def test_opt_in_cache_isolation_and_fixed_split(self):
        with tempfile.TemporaryDirectory() as directory, patch('datasets.shd.SHDWrapper', FakeSHD):
            def dataset(**extra):
                return SHDLDM(data_path=directory, spatial_factor=.2, window_size=4, **extra)
            legacy = dataset(validate_on='test')
            first = dataset(validate_on=.2, random_seed=42, cache_namespace='controlled_v2')
            second = dataset(validate_on=.2, random_seed=42, cache_namespace='controlled_v2')
            changed_seed = dataset(validate_on=.2, random_seed=123, cache_namespace='controlled_v2')
            changed_split = dataset(validate_on=.1, random_seed=42, cache_namespace='controlled_v2')
            changed_frames = SHDLDM(data_path=directory, spatial_factor=.2, window_size=8,
                                   validate_on=.2, cache_namespace='controlled_v2')
            self.assertEqual(Path(legacy.cache_path), Path(directory)/'cache/SHD')
            self.assertEqual(first.cache_path, second.cache_path)
            for other in (legacy, changed_seed, changed_split, changed_frames):
                self.assertNotEqual(first.cache_path, other.cache_path)
            self.assertEqual(first.data_train.indices, second.data_train.indices)
            self.assertEqual(first.data_val.indices, second.data_val.indices)
            self.assertEqual(len(first.data_train), 80)
            self.assertEqual(len(first.data_val), 20)
            self.assertFalse(set(first.data_train.indices) & set(first.data_val.indices))
            self.assertFalse(first.data_test.train)
            self.assertTrue(first.data_val.dataset.train)
            self.assertFalse(Path(first.cache_path).exists())
            with self.assertRaises(ValueError):
                dataset(cache_namespace='../unsafe')


if __name__ == '__main__':
    unittest.main()
