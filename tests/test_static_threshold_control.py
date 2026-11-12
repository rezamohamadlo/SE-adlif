"""Static-control forward/config/gradient checks; no optimizer or Trainer.fit."""
import contextlib
import copy
import hashlib
import io
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from models.ours.dth_adlif import DTHSEAdLIF
from models.ours.static_threshold_adlif import StaticThresholdSEAdLIF
from models.pl_module import build_training_model
from module.study_control import load_study, study_overrides, validate_config, forward_identity
from project_paths import ROOT


STUDY = 'config/studies/complementary_C002_static_control_v1.yaml'


def resolved(model):
    study = load_study(STUDY)
    with patch('datasets.shd_speaker_split.load_manifest',
               return_value=dict(validation_speaker_ids=[0, 3], role='development')):
        overrides = study_overrides(study, 30, model, 'SHD', 42)
        with initialize_config_dir(config_dir=str(ROOT/'config'), version_base=None):
            cfg = compose(config_name='main', overrides=[f'experiment=SHD_{model}',
                '+comparison=SHD', f'++model_variant={model}', 'early_stopping=false']+overrides)
        cfg = OmegaConf.to_container(cfg, resolve=True)
        validate_config(cfg)
    return cfg


class StaticThresholdControlTests(unittest.TestCase):
    def test_frozen_zero_gains_and_static_threshold_independent_of_traces(self):
        cfg = OmegaConf.create(resolved('STATIC_LS_SE_adLIF')['l1'])
        cfg.input_size = 4
        cfg.n_neurons = 5
        devices = ['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                layer = StaticThresholdSEAdLIF(cfg).to(device)
                for gain in (layer.homeostatic_gain, layer.transient_gain):
                    self.assertFalse(gain.requires_grad)
                    self.assertEqual(float(gain), 0.)
                with torch.no_grad():
                    layer.threshold_logits.copy_(torch.linspace(-3, 3, 5, device=device))
                fast = torch.rand(2, 5, device=device, requires_grad=True)
                slow = torch.rand(2, 5, device=device, requires_grad=True)
                threshold = layer.dynamic_threshold(fast, slow)
                expected = .5+layer.threshold_logits.sigmoid()
                torch.testing.assert_close(threshold, expected.expand_as(threshold), rtol=0, atol=0)
                torch.testing.assert_close(layer.dynamic_threshold(fast*10, slow-3), threshold, rtol=0, atol=0)
                self.assertTrue(((threshold > .5)&(threshold < 1.5)).all())
                for grad in torch.autograd.grad(threshold.sum(), (fast, slow)):
                    torch.testing.assert_close(grad, torch.zeros_like(grad), rtol=0, atol=0)

    def test_matches_zero_gain_dth_forward_and_common_parameter_gradients(self):
        cfg = OmegaConf.create(resolved('STATIC_LS_SE_adLIF')['l1'])
        cfg.input_size = 4
        cfg.n_neurons = 5
        devices = ['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                torch.manual_seed(117)
                dynamic = DTHSEAdLIF(cfg).to(device)
                static = StaticThresholdSEAdLIF(cfg).to(device)
                static.load_state_dict(dynamic.state_dict(), strict=True)
                x = torch.rand(2, 20, 4, device=device)*10
                a = dynamic.layer_forward(x)
                b = static.layer_forward(x)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.sum().backward()
                b.sum().backward()
                for name in ('threshold_logits', 'weight', 'recurrent', 'a', 'b'):
                    grad = getattr(static, name).grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                    torch.testing.assert_close(grad, getattr(dynamic, name).grad, rtol=1e-5, atol=1e-6)
                self.assertGreater(int(static.threshold_logits.grad.count_nonzero()), 0)
                self.assertIsNone(static.homeostatic_gain.grad)
                self.assertIsNone(static.transient_gain.grad)

    def test_rejects_nonzero_or_nonfinite_activity_gain_checkpoints(self):
        cfg = OmegaConf.create(resolved('STATIC_LS_SE_adLIF')['l1'])
        cfg.input_size = 4
        cfg.n_neurons = 5
        for name in ('homeostatic_gain', 'transient_gain'):
            for value in (.01, float('nan')):
                layer = StaticThresholdSEAdLIF(cfg)
                state = copy.deepcopy(layer.state_dict())
                state[name].fill_(value)
                with self.subTest(name=name, value=value), self.assertRaisesRegex(RuntimeError, 'requires zero'):
                    layer.load_state_dict(state, strict=True)

    def test_model_factory_preserves_rng_state_and_initial_eval_predictions(self):
        configs = [OmegaConf.create(resolved(model)) for model in
                   ('DTH_LS_SE_adLIF', 'STATIC_LS_SE_adLIF')]
        for cfg in configs:
            cfg.l1.input_size = 4
            cfg.l1.n_neurons = 5
            cfg.l2.input_size = 5
            cfg.l2.n_neurons = 5
            cfg.l_out.input_size = 5
            cfg.l_out.n_neurons = 3
            cfg.dataset.num_classes = 3
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(42)
            dynamic = build_training_model(configs[0])
            rng = torch.get_rng_state().clone()
            torch.manual_seed(42)
            static = build_training_model(configs[1])
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertIsInstance(static.l1, StaticThresholdSEAdLIF)
            self.assertIsInstance(static.l2, StaticThresholdSEAdLIF)
            self.assertEqual(set(dynamic.state_dict()), set(static.state_dict()))
            for name, value in dynamic.state_dict().items():
                torch.testing.assert_close(value, static.state_dict()[name], rtol=0, atol=0)
            total = lambda model: sum(p.numel() for p in model.parameters())
            trainable = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(total(dynamic), total(static))
            self.assertEqual(trainable(dynamic)-trainable(static), 4)
            dynamic.eval()
            static.eval()
            x = torch.rand(2, 20, 4)*4
            with torch.no_grad():
                torch.testing.assert_close(dynamic(x), static(x), rtol=0, atol=0)

    def test_matched_configs_and_frozen_cell_guard(self):
        configs = [resolved(model) for model in load_study(STUDY)['models']]
        for cfg in configs:
            self.assertEqual(cfg['dataset'], configs[0]['dataset'])
            for key in ('lr', 'n_epochs', 'batch_size', 'dropout', 'training_objective',
                        'label_smoothing_epsilon', 'gradient_clip_val', 'scheduler', 'patience', 'factor'):
                self.assertEqual(cfg[key], configs[0][key])
            self.assertEqual(cfg['n_epochs'], 300)
            self.assertEqual(cfg['stage_ceiling'], 30)
            self.assertFalse(cfg['run_test'])
        invalid = copy.deepcopy(configs[1])
        invalid['l2']['cell'] = 'dth_se_adlif'
        with self.assertRaisesRegex(ValueError, 'neuron'):
            validate_config(invalid)
        name = 'models/ours/static_threshold_adlif.py'
        expected = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        self.assertEqual(forward_identity('SHD')[name], expected)


if __name__ == '__main__':
    unittest.main()
