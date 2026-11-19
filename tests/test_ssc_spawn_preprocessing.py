"""Exact data/mask parity and spawn serialization; no model training."""
import pickle
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import h5py
import numpy as np
import torch

from datasets.ssc import SSCLDM


class SSCSpawnPreprocessingTests(unittest.TestCase):
    def test_opt_in_matches_legacy_for_short_and_long_events_and_survives_pickle(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy=SSCLDM(tmp, num_workers=0, validation_only_data=True)
            spawned=SSCLDM(tmp, num_workers=2, validation_only_data=True, spawn_safe_transforms=True)
            restored=pickle.loads(pickle.dumps(spawned.static_data_transform))
            dtype=np.dtype([('x',np.int16),('t',np.int64),('p',np.int16)])
            for duration in (10000,1500000):
                events=np.zeros(100,dtype=dtype)
                events['x']=np.arange(100)%700
                events['t']=np.linspace(0,duration,100).astype(np.int64)
                events['p']=1
                expected=legacy.static_data_transform(events.copy())
                torch.testing.assert_close(spawned.static_data_transform(events.copy()),expected,rtol=0,atol=0)
                torch.testing.assert_close(restored(events.copy()),expected,rtol=0,atol=0)

    def test_real_wrapper_roundtrip_retains_labels_and_ignore_mask(self):
        from datasets.ssc import SSCWrapper
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'SSCWrapper';root.mkdir()
            with h5py.File(root/'ssc_train.h5','w') as f:
                f['labels']=[7]
                f['extra/keys']=np.arange(35)
                f['extra/speaker']=[2]
                times=f.create_dataset('spikes/times',(1,),dtype=h5py.vlen_dtype(np.dtype('float16')))
                units=f.create_dataset('spikes/units',(1,),dtype=h5py.vlen_dtype(np.dtype('int16')))
                times[0]=np.array([0,.02,.04,.7],dtype=np.float16)
                units[0]=np.array([0,10,200,699],dtype=np.int16)
            a=SSCLDM(tmp,validation_only_data=True)
            b=SSCLDM(tmp,validation_only_data=True,spawn_safe_transforms=True)
            # Tonic's availability check expects a full release, not this tiny
            # isolated fixture. Never let a unit test trigger a download.
            with patch.object(SSCWrapper,'_check_exists',return_value=True):
                first=SSCWrapper(tmp,transform=a.static_data_transform)
                second=pickle.loads(pickle.dumps(SSCWrapper(tmp,transform=b.static_data_transform)))
            for x,y in zip(first[0],second[0]):
                torch.testing.assert_close(torch.as_tensor(x),torch.as_tensor(y),rtol=0,atol=0)
            self.assertEqual(second[0][1],7)
            self.assertTrue((second[0][2][:10]==0).all())
            self.assertTrue((second[0][2][10:]==1).all())


if __name__=='__main__':
    unittest.main()
