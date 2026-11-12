"""Train-file-only split, exposure and clean-validation controls; never fit models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import torch
from datasets.shd_speaker_split import build_development_manifest,save_manifest,load_manifest,DEVELOPMENT_PROTOCOL
from functional.count_thinning import thin_input_counts
from module.study_control import validate_resume_checkpoint


class SpeakerDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);p=self.root/'SHDWrapper';p.mkdir()
        speakers=np.repeat([0,1,2,3,6,7,8,9,10,11],20)
        labels=np.tile(np.arange(20),10)
        with h5py.File(p/'shd_train.h5','w') as f:
            f['labels']=labels;f['extra/speaker']=speakers;f['extra/keys']=np.arange(20)
            f['spikes/times']=np.zeros(len(labels));f['spikes/units']=np.zeros(len(labels))

    def test_training_only_and_exact_frozen_indices(self):
        m=build_development_manifest(self.root,[0,3],'development')
        self.assertFalse((self.root/'SHDWrapper/shd_test.h5').exists())
        self.assertFalse(m['official_test_opened']);self.assertNotIn('test_sha256',m['dataset_identity'])
        self.assertEqual(m['subsets']['train']['speaker_ids'],[1,2,6,7,8,9,10,11])
        train=set(m['subsets']['train']['indices']);val=set(m['subsets']['validation']['indices'])
        self.assertFalse(train&val);self.assertEqual(train|val,set(range(200)))
        self.assertTrue(all(m['subsets'][s]['class_counts'][c]>0 for s in ['train','validation'] for c in range(20)))
        p=self.root/'split.json';save_manifest(p,m)
        self.assertEqual(load_manifest(p,self.root,m['fingerprint']),m)

    def test_second_pair_is_distinct_and_does_not_reuse_primary_weights(self):
        a=build_development_manifest(self.root,[0,3],'development')
        b=build_development_manifest(self.root,[6,10],'pair_sensitivity')
        self.assertNotEqual(a['fingerprint'],b['fingerprint'])
        self.assertTrue(set(b['subsets']['validation']['indices']).issubset(a['subsets']['train']['indices']))

    def test_missing_class_or_changed_pair_fails_without_fallback(self):
        with h5py.File(self.root/'SHDWrapper/shd_train.h5','a') as f:
            y=f['labels'][:];s=f['extra/speaker'][:];y[np.isin(s,[0,3])&(y==19)]=18;f['labels'][:]=y
        with self.assertRaises(ValueError):build_development_manifest(self.root,[0,3],'development')

    def test_datamodule_never_constructs_test_dataset(self):
        from datasets.shd import SHDLDM
        from datasets.shd_speaker_split import read_metadata
        m=build_development_manifest(self.root,[0,3],'development');p=self.root/'split.json';save_manifest(p,m)
        class TrainOnlyWrapper(torch.utils.data.Dataset):
            sensor_size=(700,1,1)
            def __init__(inner,save_to,train=True,**kwargs):
                if train is not True:raise AssertionError('Official-test dataset constructed')
                inner.location_on_system=str(Path(save_to)/'SHDWrapper');inner.data_filename='shd_train.h5'
                inner.speaker,inner.labels=read_metadata(Path(inner.location_on_system)/inner.data_filename)
            def __len__(inner):return len(inner.labels)
        with patch('datasets.shd.SHDWrapper',TrainOnlyWrapper):
            dm=SHDLDM(str(self.root),validate_on='speaker_holdout',cache_namespace=DEVELOPMENT_PROTOCOL,
                      speaker_split_manifest=str(p),speaker_split_fingerprint=m['fingerprint'],
                      validation_only_data=True,random_seed=42)
        self.assertFalse(hasattr(dm,'data_test'));self.assertEqual(dm.data_train.indices,m['subsets']['train']['indices'])

    def test_resume_cannot_import_a_checkpoint_that_saw_held_out_speakers(self):
        root=self.root/'seed';(root/'ckpt').mkdir(parents=True);path=root/'ckpt/last.ckpt'
        cfg=dict(logdir=str(root),ckpt_path=str(path),stage_ceiling=30,dataset=dict(validate_on='speaker_holdout',speaker_split_fingerprint='new'))
        old=dict(cfg,dataset=dict(validate_on=.2,speaker_split_fingerprint=None))
        torch.save(dict(hyper_parameters=dict(cfg=old)),path)
        with self.assertRaises(ValueError):validate_resume_checkpoint(cfg)
        torch.save(dict(hyper_parameters=dict(cfg=cfg)),path);validate_resume_checkpoint(cfg)


class CountDeletionTests(unittest.TestCase):
    def test_clean_evaluation_and_zero_probability_are_exact_identities(self):
        x=torch.tensor([[[0.,3.],[-1.,0.],[0.,0.]]]);state=torch.random.get_rng_state()
        self.assertIs(thin_input_counts(x,.1,False),x)
        self.assertIs(thin_input_counts(x,0,True),x)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))

    def test_per_event_deletion_preserves_support_padding_and_markers(self):
        x=torch.full((2,200,3),20.);x[:,-10:]=0;x[0,-1,0]=-1;original=x.clone()
        torch.manual_seed(42);a=thin_input_counts(x,.1,True);b=thin_input_counts(x,.1,True)
        self.assertEqual(a.shape,x.shape);self.assertTrue(torch.equal(x,original))
        self.assertTrue((a[x==0]==0).all());self.assertEqual(a[0,-1,0],-1)
        self.assertTrue(((a>0)&(a<20)).any());self.assertFalse(torch.equal(a,b))
        self.assertTrue((thin_input_counts(x,1,True)[x>=0]==0).all())

    def test_fractional_currents_are_not_treated_as_event_counts(self):
        with self.assertRaises(ValueError):thin_input_counts(torch.tensor([.5]),.1,True)


if __name__=='__main__':unittest.main()
