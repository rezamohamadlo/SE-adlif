"""Existing validation-only adapters, native event domains and saved target masks."""
import numpy as np
import torch

from datasets.utils.pad_tensors import PadTensors
from scripts.adaptation_analysis.shd_data import SHDValidationData, preflight_shd, _select
from scripts.adaptation_ssc.data import metadata as ssc_metadata, SSCValidationData
from scripts.adaptation_ecg.data import load_validation
from scripts.adaptation_gradient_pilot.methods import require


class Adapter:
    def __init__(self,dataset,source,cfg,baseline,spec):
        self.dataset=dataset;self.spec=spec
        if dataset=='SHD':
            self.meta=preflight_shd(cfg);require(self.meta==source['data'],'SHD* metadata differs')
            self.ids=list(source['sample_ids']);self.native=SHDValidationData(cfg,sample_ids=self.ids,verify_hash=False)
        elif dataset=='SSC':
            self.meta,labels,speakers=ssc_metadata(cfg,source['specification'],baseline)
            require(self.meta==source['data'],'SSC metadata differs');self.ids=self.meta['validation_indices']
            self.native=SSCValidationData(cfg,self.meta,labels,speakers,self.ids)
        else:
            require(dataset=='ECG','Unknown dataset')
            self.native,self.meta=load_validation(source['specification'],baseline,cfg)
            require(self.meta==source['data'],'ECG metadata differs');self.ids=list(self.native.ids)
        self.selected=(list(self.ids) if dataset=='ECG' else
            _select(self.native.labels,self.ids,spec['samples'][dataset],spec['cohort_seed']))
        self.repeat_ids=self.selected[:spec['repeat_records']]
        self.diagnostic_ids=self.selected[:spec['diagnostic_records']]
        self.domain='signed' if dataset=='ECG' else 'counts'
        self.burn=0 if dataset=='ECG' else self.meta['ignore_first_timesteps']
        self.bin_ms=None if dataset=='ECG' else self.meta['bin_ms']

    def input(self,sid):
        require(sid in self.ids,'Non-validation ID')
        return (self.native.x[sid].astype(np.float32,copy=True) if self.dataset=='ECG' else
                self.native.load(sid)['counts'].astype(np.float32))

    def item(self,sid,x=None):
        if x is None:x=self.input(sid)
        if self.dataset!='ECG':return self.native.model_item(sid,x)
        require(np.isin(x,[-1,0,1]).all(),'ECG signed event domain differs')
        return torch.from_numpy(np.asarray(x,dtype=np.float32).copy()),torch.from_numpy(self.native.y[sid]).long(),torch.arange(1,len(x)+1,dtype=torch.long)

    def batch(self,ids,override=None):
        return PadTensors()([self.item(i,None if override is None else override[i]) for i in ids])
