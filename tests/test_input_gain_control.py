"""Analytical input-control checks; no training, optimizer or real dataset."""
import unittest
import contextlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from models.ours.input_fir import CausalInputFIR
from models.pl_module import build_training_model
from module.study_control import load_study,study_overrides,validate_config,finish_stage
from project_paths import ROOT


class InputGainControlTests(unittest.TestCase):
    def test_identity_initialization_and_parameter_match(self):
        causal=CausalInputFIR(2,[1,2,4,8]);gain=CausalInputFIR(2,[1,2,4,8],mode='gain_only')
        x=torch.randn(3,6,2)
        torch.testing.assert_close(gain(x),x,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in causal.parameters()),sum(p.numel() for p in gain.parameters()))

    def test_no_history_mixing_and_analytical_gradient(self):
        gain=CausalInputFIR(2,[1,3],mode='gain_only')
        with torch.no_grad():gain.coefficients.copy_(torch.tensor([[.2,-.1],[.3,.4]]))
        x=torch.arange(20,dtype=torch.float32).reshape(2,5,2)
        scale=1+gain.coefficients.tanh().sum(0)
        torch.testing.assert_close(gain(x),x*scale)
        gain(x).sum().backward()
        expected=(1-gain.coefficients.detach().tanh().square())*x.sum((0,1))
        torch.testing.assert_close(gain.coefficients.grad,expected)
        impulse=torch.zeros(1,5,2);impulse[:,0]=1
        self.assertTrue((gain(impulse)[:,1:]==0).all())

    def test_causal_math_and_gradient_remain_unchanged(self):
        causal=CausalInputFIR(2,[1,3]);x=torch.arange(20,dtype=torch.float32).reshape(2,5,2)
        with torch.no_grad():causal.coefficients.copy_(torch.tensor([[.2,-.1],[.3,.4]]))
        expected=x.clone()
        for t in range(5):
            for lag,c in zip([1,3],causal.coefficients.tanh()):
                if t>=lag:expected[:,t]=expected[:,t]+c*x[:,t-lag]
        torch.testing.assert_close(causal(x),expected,rtol=0,atol=0)
        causal(x).sum().backward()
        grad=torch.stack([(1-causal.coefficients.detach()[i].tanh().square())*x[:,:5-lag].sum((0,1)) for i,lag in enumerate([1,3])])
        torch.testing.assert_close(causal.coefficients.grad,grad)

    def test_unknown_mode_is_not_silently_treated_as_causal(self):
        with self.assertRaises(ValueError):CausalInputFIR(2,[1],mode='typo')

    def test_config_inheritance_and_declared_mode_guard(self):
        study=load_study('config/studies/complementary_gain_control_A001_v1.yaml')
        frozen=dict(validation_speaker_ids=[0,3],role='development')
        with patch('datasets.shd_speaker_split.load_manifest',return_value=frozen):
            overrides=study_overrides(study,30,'GAIN_SE_adLIF','SHD',42)
            with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
                cfg=compose(config_name='main',overrides=['experiment=SHD_GAIN_SE_adLIF','+comparison=SHD','++model_variant=GAIN_SE_adLIF']+overrides)
            resolved=OmegaConf.to_container(cfg,resolve=True);validate_config(resolved)
            self.assertEqual(resolved['n_epochs'],300);self.assertEqual(resolved['input_fir']['mode'],'gain_only')
            resolved['input_fir']['mode']='causal'
            with self.assertRaisesRegex(ValueError,'mechanism control'):validate_config(resolved)
        # Full classifier initialization and RNG consumption also match.
        causal_cfg=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        causal_cfg.input_fir.mode='causal'
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(42);gain_model=build_training_model(cfg);gain_rng=torch.get_rng_state()
            torch.manual_seed(42);causal_model=build_training_model(causal_cfg);causal_rng=torch.get_rng_state()
        self.assertEqual(gain_model.input_fir.mode,'gain_only')
        self.assertTrue(torch.equal(gain_rng,causal_rng))
        for key,value in gain_model.state_dict().items():
            torch.testing.assert_close(value,causal_model.state_dict()[key],rtol=0,atol=0)

    def test_gain_control_accounting_includes_shared_parameter_work(self):
        from measure_reference_costs import input_fir_operations
        gain=CausalInputFIR(2,[1,3],mode='gain_only')
        operations=input_fir_operations(SimpleNamespace(input_fir=gain),torch.ones(3,6,2))
        self.assertEqual(operations,{'tanh':4,'sum':2,'add':2,'mul':36})

    def test_stage_report_distinguishes_short_and_full_budget_without_training(self):
        # Presentation/receipt fixture only: never instantiate or fit a Trainer.
        for budget in (1,2):
            with self.subTest(budget=budget),tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp)/'seed';(d/'ckpt').mkdir(parents=True)
                (Path(tmp)/'study').mkdir()
                for name in ('best','last'):torch.save({'epoch':0},d/'ckpt'/f'{name}.ckpt')
                logs=d/'logs/mlp_snn/version_0';logs.mkdir(parents=True)
                (logs/'metrics.csv').write_text('epoch,val_acc\n0,0.5\n')
                (d/'STATUS.json').write_text('{"status":"running"}')
                cfg=OmegaConf.create(dict(logdir=str(d),stage_ceiling=1,n_epochs=budget,model_variant='FIR_SE_adLIF',
                    paper_baseline_dataset='SHD',random_seed=42,evaluation_protocol='unit_fixture',
                    dataset=dict(validate_on='speaker_holdout',speaker_split_fingerprint='fixture'),study_file='fixture.yaml'))
                trainer=SimpleNamespace(checkpoint_callback=SimpleNamespace(best_model_path=str(d/'ckpt/best.ckpt'),best_model_score=torch.tensor(.5)))
                with patch('module.study_control.validate_config'), \
                     patch('module.study_control.load_study',return_value={'output_root':str(Path(tmp)/'study/run')}), \
                     contextlib.redirect_stdout(io.StringIO()):
                    finish_stage(cfg,trainer)
                text=(d/'TRAINING_SUMMARY.md').read_text()
                receipt=json.loads((d/'stages/001/RESULT.json').read_text())
                self.assertEqual(receipt['preliminary'],budget>1)
                if budget==1:self.assertIn('full training budget is complete',text)
                else:self.assertIn('preliminary screen',text)
                self.assertIn('No official test evaluation',text)


if __name__=='__main__':unittest.main()
