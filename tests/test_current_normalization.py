"""Current-conditioning math/gradient/config checks; no fit or optimizer steps."""
import contextlib
import copy
import io
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from models.alif import SEAdLIF
from models.ours.current_norm_adlif import ResidualCurrentNormSEAdLIF
from models.pl_module import build_training_model
from module.initial_state import state_fingerprint
from module.study_control import load_study, study_overrides, validate_config
from project_paths import ROOT


STUDY = 'config/studies/complementary_C004_current_norm_v1.yaml'


def cell_config(mode='layer_norm'):
    return OmegaConf.create(dict(input_size=4, n_neurons=6, tau_u_range=[5,25],
        tau_w_range=[60,300], q=120, current_norm_mode=mode, current_norm_eps=1e-5,
        cell='residual_current_norm_se_adlif' if mode=='layer_norm' else 'current_gain_se_adlif'))


def configured(model):
    study = load_study(STUDY)
    with patch('datasets.shd_speaker_split.load_manifest',
               return_value=dict(validation_speaker_ids=[0,3],role='development')):
        overrides = study_overrides(study,30,model,'SHD',42)
        with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
            cfg = compose(config_name='main',overrides=[f'experiment=SHD_{model}',
                '+comparison=SHD',f'++model_variant={model}','early_stopping=false']+overrides)
        cfg = OmegaConf.to_container(cfg,resolve=True)
        validate_config(cfg)
    return cfg


class CurrentNormalizationTests(unittest.TestCase):
    def test_zero_gate_reference_limit_step_scan_states_cpu_and_cuda(self):
        devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            for mode in ('layer_norm','gain_only'):
                with self.subTest(device=device,mode=mode):
                    cfg=cell_config(mode)
                    base=SEAdLIF(cfg).to(device)
                    variant=ResidualCurrentNormSEAdLIF(cfg).to(device)
                    missing,unexpected=variant.load_state_dict(base.state_dict(),strict=False)
                    self.assertEqual(missing,['current_norm_gate'])
                    self.assertEqual(unexpected,[])
                    x=torch.randn(2,20,4,device=device)*4
                    torch.testing.assert_close(base.layer_forward(x),variant.layer_forward(x),rtol=0,atol=0)
                    a=base.initial_state(2,device)
                    b=variant.initial_state(2,device)
                    for frame in x.unbind(1):
                        ao,a=base(frame,a)
                        bo,b=variant(frame,b)
                        torch.testing.assert_close(ao,bo,rtol=0,atol=0)
                        for first,second in zip(a,b):
                            torch.testing.assert_close(first,second,rtol=0,atol=0)
                    astates,aout=base.layer_forward_with_states(x)
                    bstates,bout=variant.layer_forward_with_states(x)
                    torch.testing.assert_close(astates,bstates,rtol=0,atol=0)
                    torch.testing.assert_close(aout,bout,rtol=0,atol=0)

    def test_normalized_current_matches_manual_oracle_and_gradients(self):
        devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                layer=ResidualCurrentNormSEAdLIF(cell_config()).to(device)
                with torch.no_grad():
                    layer.current_norm_gate.fill_(.4)
                    layer.bias.copy_(torch.linspace(-.3,.3,6,device=device))
                x=torch.randn(3,5,4,device=device)*2
                actual=layer.feedforward_current(x)
                c=F.linear(x,layer.weight,layer.bias)
                d=c-layer.bias
                normalized=(d-d.mean(-1,keepdim=True))*torch.rsqrt(d.var(-1,unbiased=False,keepdim=True)+1e-5)
                expected=c+layer.current_norm_gate.tanh()*normalized
                torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
                weights=torch.linspace(.5,2.,6,device=device)
                parameters=(layer.weight,layer.bias,layer.current_norm_gate)
                a=torch.autograd.grad((actual*weights).sum(),parameters,retain_graph=True)
                b=torch.autograd.grad((expected*weights).sum(),parameters)
                for first,second in zip(a,b):
                    self.assertTrue(torch.isfinite(first).all())
                    torch.testing.assert_close(first,second,rtol=3e-5,atol=1e-5)

    def test_no_batch_or_future_pooling_and_zero_input_preserves_bias(self):
        for mode in ('layer_norm','gain_only'):
            with self.subTest(mode=mode):
                layer=ResidualCurrentNormSEAdLIF(cell_config(mode))
                with torch.no_grad():
                    layer.current_norm_gate.fill_(.7)
                    layer.bias.copy_(torch.linspace(-.5,.5,6))
                x=torch.randn(2,8,4)
                changed=x.clone()
                changed[1]*=20
                changed[0,4:]+=5
                torch.testing.assert_close(layer.feedforward_current(x)[0,:4],
                    layer.feedforward_current(changed)[0,:4],rtol=0,atol=0)
                torch.testing.assert_close(layer.layer_forward(x)[0,:4],
                    layer.layer_forward(changed)[0,:4],rtol=0,atol=0)
                quiet=layer.feedforward_current(torch.zeros(2,8,4))
                torch.testing.assert_close(quiet,layer.bias.expand_as(quiet),rtol=0,atol=0)

    def test_gain_control_is_parameter_matched_and_analytically_rescales_only(self):
        norm=ResidualCurrentNormSEAdLIF(cell_config())
        gain=ResidualCurrentNormSEAdLIF(cell_config('gain_only'))
        gain.load_state_dict(norm.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in norm.parameters()),sum(p.numel() for p in gain.parameters()))
        with torch.no_grad():
            gain.current_norm_gate.fill_(.5)
            norm.current_norm_gate.fill_(.5)
        x=torch.randn(2,7,4)
        expected=F.linear(x,gain.weight,gain.bias)+gain.current_norm_gate.tanh()*F.linear(x,gain.weight,None)
        torch.testing.assert_close(gain.feedforward_current(x),expected,rtol=1e-5,atol=1e-6)
        self.assertFalse(torch.allclose(norm.feedforward_current(x),gain.feedforward_current(x)))

    def test_zero_gate_has_immediate_finite_gradient_cpu_cuda(self):
        devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            for mode in ('layer_norm','gain_only'):
                with self.subTest(device=device,mode=mode):
                    torch.manual_seed(42)
                    layer=ResidualCurrentNormSEAdLIF(cell_config(mode)).to(device)
                    layer.layer_forward(torch.rand(2,20,4,device=device)*8).sum().backward()
                    self.assertIsNotNone(layer.current_norm_gate.grad)
                    self.assertTrue(torch.isfinite(layer.current_norm_gate.grad))
                    self.assertGreater(float(layer.current_norm_gate.grad.abs()),0)
                    self.assertTrue(torch.isfinite(layer.weight.grad).all())
                    with torch.no_grad():layer.current_norm_gate.fill_(1.)
                    layer.reset_parameters()
                    self.assertEqual(float(layer.current_norm_gate),0.)

    def test_configs_initial_common_tensors_and_predeclared_controls(self):
        study=load_study(STUDY)
        self.assertEqual(study['initial_screen_models'],study['models'][:4])
        self.assertEqual(study['conditional_gain_controls']['RCN_LS_SE_adLIF'],'CG_LS_SE_adLIF')
        fingerprints={}
        for model in study['models']:
            cfg=OmegaConf.create(configured(model))
            self.assertEqual(cfg.lr,.01)
            self.assertEqual(cfg.n_epochs,300)
            self.assertEqual(cfg.batch_size,1024)
            self.assertEqual(cfg.dropout,.15)
            self.assertFalse(cfg.run_test)
            self.assertTrue(cfg.study_record_initial_state)
            with contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(42)
                instance=build_training_model(cfg)
            fingerprints[model]=state_fingerprint(instance)
        reference=fingerprints['SE_adLIF']
        for name,item in reference['tensors'].items():
            for model in fingerprints:
                self.assertEqual(item['sha256'],fingerprints[model]['tensors'][name]['sha256'])
        self.assertEqual(reference['model_state_sha256'],fingerprints['LS_SE_adLIF']['model_state_sha256'])
        expected=fingerprints['RCN_SE_adLIF']['model_state_sha256']
        for model in ('RCN_LS_SE_adLIF','CG_SE_adLIF','CG_LS_SE_adLIF'):
            self.assertEqual(expected,fingerprints[model]['model_state_sha256'])
        invalid=configured('RCN_SE_adLIF')
        invalid['l1']['current_norm_eps']=1e-4
        with self.assertRaisesRegex(ValueError,'neuron setting'):
            validate_config(invalid)

    def test_smoothing_integration_keeps_ordinary_eval_ce_and_original_masks(self):
        cfg=OmegaConf.create(configured('RCN_LS_SE_adLIF'))
        cfg.l1.input_size=4;cfg.l1.n_neurons=6;cfg.l2.input_size=6;cfg.l2.n_neurons=6
        cfg.l_out.input_size=6;cfg.l_out.n_neurons=3;cfg.dataset.num_classes=3
        with contextlib.redirect_stdout(io.StringIO()):model=build_training_model(cfg)
        with torch.no_grad():
            model.l1.current_norm_gate.fill_(.3);model.l2.current_norm_gate.fill_(.3)
        x=torch.randn(2,20,4);targets=torch.tensor([[-1,0],[-1,2]],dtype=torch.int32)
        blocks=torch.ones(2,20,dtype=torch.long);blocks[:,:2]=0;valid=targets.flatten().ne(-1)
        model.eval();reduced,loss,_=model.process_predictions_and_compute_losses(model(x),targets,blocks)
        torch.testing.assert_close(loss,F.cross_entropy(reduced[valid].float(),targets.flatten()[valid].long()),rtol=0,atol=0)
        model.train();reduced,loss,_=model.process_predictions_and_compute_losses(model(x),targets,blocks)
        torch.testing.assert_close(loss,F.cross_entropy(reduced[valid].float(),targets.flatten()[valid].long(),label_smoothing=.1))
        loss.backward()
        self.assertTrue(torch.isfinite(model.l1.current_norm_gate.grad))
        self.assertTrue(torch.isfinite(model.l2.current_norm_gate.grad))
        self.assertEqual(targets.dtype,torch.int32)

    def test_invalid_modes_or_epsilon_rejected(self):
        for epsilon in (True,0,-1,float('nan'),float('inf'),'1e-5'):
            cfg=cell_config();cfg.current_norm_eps=epsilon
            with self.subTest(epsilon=epsilon),self.assertRaisesRegex(ValueError,'epsilon'):
                ResidualCurrentNormSEAdLIF(cfg)
        cfg=cell_config();cfg.current_norm_mode='gain_only'
        with self.assertRaisesRegex(ValueError,'registered cell'):
            ResidualCurrentNormSEAdLIF(cfg)


if __name__=='__main__':unittest.main()
