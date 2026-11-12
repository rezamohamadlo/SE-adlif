"""CPU correctness checks; no Trainer.fit or main-experiment initialization."""
import contextlib
import copy
import io
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from models.ours.input_fir import CausalInputFIR, InputFIRMLPSNN
from models.pl_module import MLPSNN
from module.study_control import canonical, load_study, study_overrides, validate_config
from project_paths import ROOT


class FIRTests(unittest.TestCase):
    def test_identity_and_immediate_channel_gradients(self):
        module=CausalInputFIR(3,[1,2,4,8])
        x=torch.arange(36,dtype=torch.float32).reshape(2,6,3)
        torch.testing.assert_close(module(x),x,rtol=0,atol=0)
        module(x).sum().backward()
        self.assertTrue(torch.isfinite(module.coefficients.grad).all())
        self.assertTrue((module.coefficients.grad[:3].abs().sum(-1)>0).all())

    def test_causal_and_channel_specific(self):
        module=CausalInputFIR(2,[1,2])
        with torch.no_grad():module.coefficients[:,0]=.3
        x=torch.ones(1,8,2); y=x.clone();y[:,5:]*=9
        torch.testing.assert_close(module(x)[:,:5],module(y)[:,:5],rtol=0,atol=0)
        torch.testing.assert_close(module(x)[...,1],x[...,1],rtol=0,atol=0)
        self.assertEqual(module(x[:,:1]).shape,(1,1,2))

    def test_exact_reference_network_limit_and_finite_gradients(self):
        with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
            cfg=compose(config_name='main',overrides=['experiment=SHD_FIR_SE_adLIF','+comparison=SHD'])
        cfg.l1.input_size=3;cfg.l1.n_neurons=5;cfg.l2.input_size=5;cfg.l2.n_neurons=5
        cfg.l_out.input_size=5;cfg.dataset.num_classes=2;cfg.dropout=0
        with contextlib.redirect_stdout(io.StringIO()):
            base=MLPSNN(cfg);variant=InputFIRMLPSNN(cfg)
        missing,unexpected=variant.load_state_dict(base.state_dict(),strict=False)
        self.assertEqual(missing,['input_fir.coefficients']);self.assertEqual(unexpected,[])
        x=torch.rand(2,25,3)*4
        base.eval();variant.eval()
        torch.testing.assert_close(base(x),variant(x),rtol=0,atol=0)
        variant(x).square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in variant.parameters() if p.grad is not None))


class StagedConfigTests(unittest.TestCase):
    def test_full_schedule_is_not_the_stage_ceiling(self):
        study=load_study('config/studies/complementary_mechanisms_v1.yaml')
        overrides=study_overrides(study,30,'FIR_SE_adLIF','SHD',42)
        with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
            cfg=compose(config_name='main',overrides=['experiment=SHD_FIR_SE_adLIF','+comparison=SHD']+overrides)
        value=OmegaConf.to_container(cfg,resolve=True)
        validate_config(value)
        self.assertEqual(value['n_epochs'],300)
        self.assertEqual(value['stage_ceiling'],30)
        self.assertFalse(value['run_test'])
        self.assertTrue(value['dataset']['validation_only_data'])
        for change in [('lr',.2),('n_epochs',30),('run_test',True)]:
            invalid=copy.deepcopy(value);invalid[change[0]]=change[1]
            with self.assertRaises(ValueError):validate_config(invalid)

    def test_only_declared_stage_and_discovery_seed(self):
        study=load_study('config/studies/complementary_mechanisms_v1.yaml')
        for ceiling,seed in [(31,42),(30,123)]:
            with self.assertRaises(ValueError):study_overrides(study,ceiling,'FIR_SE_adLIF','SHD',seed)

    def test_only_routing_and_boundary_can_change_for_full_resume(self):
        original=dict(lr=.01,n_epochs=300,stage_ceiling=30,ckpt_path=None,l1={'cell':'se_adlif'})
        later=dict(original,stage_ceiling=75,ckpt_path='results/example/last.ckpt')
        self.assertEqual(canonical(original),canonical(later))
        self.assertNotEqual(canonical(original),canonical(dict(later,lr=.02)))


class TrainingOnlyDataTests(unittest.TestCase):
    def test_ssc_opt_in_excludes_test_but_default_retains_it(self):
        from datasets.ssc import SSCLDM
        with tempfile.TemporaryDirectory() as folder:
            for enabled,expected in [(True,['train','valid']),(False,['train','test','valid'])]:
                with patch('datasets.ssc.SSCWrapper',return_value=[]) as wrapper:
                    data=SSCLDM(folder,validation_only_data=enabled,validation_batch_size=512)
                    data.setup('fit')
                    self.assertEqual([call.kwargs['split'] for call in wrapper.call_args_list],expected)
                    self.assertEqual(data.validation_batch_size,512)

    def test_ecg_opt_in_never_downloads_or_loads_official_test(self):
        import numpy as np
        from scipy.io import savemat
        from datasets.ecg import ECGLDM
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'QTDB';p.mkdir()
            savemat(p/'QTDB_train.mat',dict(x=np.zeros((20,17,4),dtype=np.int16),
                        y=np.tile(np.eye(6,dtype=np.uint8)[0],(20,17,1)),
                        t=np.tile(np.arange(17,dtype=np.uint16),(20,1))))
            with patch.object(ECGLDM,'download_if_not_exist') as download:
                data=ECGLDM(folder,valid_fraction=.2,validation_only_data=True,
                            validation_batch_size=128,batch_size=8,num_workers=0)
                self.assertEqual(list(download.call_args.args[1]),['QTDB/QTDB_train.mat'])
                data.prepare_data();data.setup('fit')
            self.assertFalse(hasattr(data,'data_test'))
            self.assertTrue(set(data.data_train.indices).isdisjoint(data.data_val.indices))
            x,y,blocks=data.data_train[0]
            self.assertEqual(x.shape,(17,4));self.assertEqual(y.shape,(17,))
            self.assertEqual(blocks.tolist(),list(range(1,18)))
            self.assertEqual(data.val_dataloader().batch_size,128)


if __name__=='__main__':unittest.main()
