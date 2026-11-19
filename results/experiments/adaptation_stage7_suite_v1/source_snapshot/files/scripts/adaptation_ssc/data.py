"""Read only the authenticated SSC validation H5; never construct Tonic datasets."""
import hashlib
import math

import h5py
import numpy as np
import torch
from tonic.datasets import SSC
from tonic.io import make_structured_array
from tonic.transforms import Downsample,ToFrame

from project_paths import local_path,portable_path
from scripts.adaptation_analysis.pilot import sha256
from scripts.adaptation_analysis.shd_data import _select


def validate_config(cfg,baseline):
    data=dict(cfg['dataset']);expected=baseline['datasets']['SSC']
    if data.get('_target_')!='datasets.ssc.SSCLDM' or data.get('num_classes')!=35:
        raise ValueError('Requires saved SSC classification dataset')
    if data.get('validate_on') is not None or data.get('additional_test_set_validation',False):
        raise ValueError('SSC requires the official validation split, without a training/test holdout')
    for key in ('spatial_factor','time_factor','window_size','pad_to_min_size','ignore_first_timesteps','batch_size'):
        if data.get(key)!=expected[key]:raise ValueError('SSC preprocessing differs: '+key)
    return data


def metadata(cfg,study,baseline):
    config=validate_config(cfg,baseline)
    path=local_path(config['data_path'])/'SSCWrapper/ssc_valid.h5'
    if portable_path(path)!=study['dataset_file']:raise ValueError('Frozen SSC validation path differs')
    if not path.is_file():raise FileNotFoundError('Transfer the recorded SSC validation file; no downloads are performed')
    checksum=sha256(path)
    if checksum!=study['dataset_sha256']:raise ValueError('SSC validation SHA-256 differs')
    with h5py.File(path,'r') as f:
        labels=np.asarray(f['labels'],dtype=np.int64)
        speakers=np.asarray(f['extra/speaker'])
        if (labels.ndim!=1 or speakers.shape!=labels.shape or len(f['spikes/times'])!=len(labels)
            or len(f['spikes/units'])!=len(labels) or len(labels)!=study['expected_validation_samples']):
            raise ValueError('SSC validation membership/metadata differs')
    if set(np.unique(labels).tolist())!=set(range(35)):raise ValueError('SSC class labels differ')
    speakers=[x.decode('utf-8') if isinstance(x,bytes) else str(x) for x in speakers]
    ids=list(range(len(labels)))
    meta=dict(path=portable_path(path),sha256=checksum,validation_samples=len(labels),
        validation_indices=ids,split='official_validation',split_seed=None,input_channels=140,
        bin_ms=config['window_size'],min_frames=config['pad_to_min_size'],ignore_first_timesteps=config['ignore_first_timesteps'],
        speaker_groups=len(set(speakers)),official_test_opened=False)
    meta['split_fingerprint']=hashlib.sha256((checksum+'|official_validation|'+str(len(ids))).encode()).hexdigest()
    return meta,labels,speakers


class SSCValidationData:
    def __init__(self,cfg,meta,labels,speakers,ids):
        self.config=cfg['dataset'];self.metadata=meta;self.path=local_path(meta['path'],must_exist=True)
        self.labels=labels;self.speakers=speakers;self.sample_ids=list(ids);self._ids=set(ids)
        if len(self._ids)!=len(ids) or not self._ids.issubset(meta['validation_indices']):raise ValueError('Invalid validation IDs')
        factor=self.config['spatial_factor'];sensor=SSC.sensor_size
        # Match SSCLDM exactly. The unused y dimension is zero for this 1D sensor;
        # Tonic's no-y framing uses polarity and x only. Do not change shared code.
        self.sensor_size=(int(math.ceil(sensor[0]*factor)),int(math.ceil(sensor[1])*factor),sensor[2])
        self.downsample=Downsample(time_factor=self.config['time_factor'],spatial_factor=factor)
        self.to_frame=ToFrame(sensor_size=self.sensor_size,time_window=self.config['window_size'])

    def __len__(self):return len(self.sample_ids)

    def load(self,sample_id):
        if sample_id not in self._ids:raise ValueError('Sample not in frozen validation cohort')
        with h5py.File(self.path,'r') as f:
            # Preserve the current saved pipeline's overflow compatibility fix.
            seconds=np.asarray(f['spikes/times'][sample_id],dtype=np.float64)
            units=np.asarray(f['spikes/units'][sample_id])
        if (seconds.shape!=units.shape or not np.isfinite(seconds).all() or np.any(seconds<0)
            or np.any(np.diff(seconds)<0) or np.any(units>=700)):
            raise ValueError('Invalid SSC raw validation events')
        if not len(seconds):raise ValueError('Empty raw record: no silent replacement of the saved framing contract')
        events=make_structured_array(seconds*1e6,units,1,dtype=SSC.dtype)
        frames=self.to_frame(self.downsample(events)).reshape(-1,140)
        counts=frames.astype(np.int64)
        if not len(counts) or np.any(counts<0):raise ValueError('Empty/overflowed SSC count frames')
        return dict(sample_id=int(sample_id),target=int(self.labels[sample_id]),speaker=self.speakers[sample_id],
            counts=counts,raw_event_count=len(seconds),framed_event_count=int(counts.sum()),
            support_frames=len(counts),stored_frames=max(len(counts),self.metadata['min_frames']),
            duration_ms=float((seconds[-1]-seconds[0])*1000),frame_support_ms=len(counts)*self.metadata['bin_ms'],
            bin_ms=self.metadata['bin_ms'],framing='Saved SSC float64-before-microseconds, Downsample/ToFrame defaults')

    def model_item(self,sample_id,counts=None):
        if sample_id not in self._ids:raise ValueError('Sample outside frozen validation IDs')
        if counts is None:counts=self.load(sample_id)['counts']
        a=np.asarray(counts)
        if a.ndim!=2 or a.shape[1]!=140 or not np.isfinite(a).all() or np.any(a<0) or not np.equal(a,np.floor(a)).all():
            raise ValueError('Invalid SSC model count input')
        x=torch.from_numpy(a.astype(np.float32,copy=True))
        if len(x)<self.metadata['min_frames']:
            x=torch.cat((x,torch.zeros(self.metadata['min_frames']-len(x),140)),0)
        blocks=torch.ones(len(x),dtype=torch.long);blocks[:self.metadata['ignore_first_timesteps']]=0
        return x,torch.tensor(int(self.labels[sample_id]),dtype=torch.long),blocks


def select_ids(labels,meta,samples,seed):
    return _select(labels,meta['validation_indices'],samples,seed)


def check_legacy_contract(data):
    """Three-record input-only preflight, no shared Dataset constructor or model."""
    from types import SimpleNamespace
    from datasets.ssc import SSCWrapper
    from datasets.utils.transforms import Flatten
    def transform(events):
        frames=torch.from_numpy(data.to_frame(data.downsample(events))).float()
        if len(frames)<data.metadata['min_frames']:
            frames=torch.cat((frames,torch.zeros(data.metadata['min_frames']-len(frames),1,140)),0)
        return Flatten()(frames)
    proxy=SimpleNamespace(location_on_system=str(data.path.parent),data_filename=data.path.name,dtype=SSC.dtype,
        transform=transform,target_transform=None,ignore_first_timesteps=data.metadata['ignore_first_timesteps'])
    for sid in (0,len(data.labels)//2,len(data.labels)-1):
        item=data.model_item(sid)
        legacy=SSCWrapper.__getitem__(proxy,sid)
        if not torch.equal(item[0],legacy[0]) or int(item[1])!=int(legacy[1]) or not torch.equal(item[2],legacy[2]):
            raise ValueError('New validation-only loader differs from saved SSC wrapper contract')
    return dict(samples_checked=[0,len(data.labels)//2,len(data.labels)-1],matches_saved_wrapper=True,
                dataset_constructor_called=False,model_forward_called=False)
