"""Config, forward, loss and gradient checks; no fit or optimizer steps."""
import contextlib
import copy
import hashlib
import io
import unittest
from unittest.mock import patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F

from models.pl_module import build_training_model
from models.ours.dth_adlif import DTHSEAdLIF
from module.study_control import load_study, study_overrides, validate_config, forward_identity
from project_paths import ROOT


STUDY = 'config/studies/complementary_C001_dth_ls_primary_v1.yaml'


def composed(model):
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


class DTHLabelSmoothingStudyTests(unittest.TestCase):
    def test_shared_conditions_and_neuron_guards(self):
        configs = [composed(model) for model in load_study(STUDY)['models']]
        for cfg in configs:
            self.assertEqual(cfg['n_epochs'], 300)
            self.assertEqual(cfg['stage_ceiling'], 30)
            self.assertEqual(cfg['batch_size'], 1024)
            self.assertEqual(cfg['dataset'], configs[0]['dataset'])
            self.assertEqual(cfg['lr'], .01)
            self.assertEqual(cfg['dropout'], .15)
            self.assertFalse(cfg['run_test'])
            self.assertIsNone(cfg.get('init_ckpt_path'))
        dth = configs[1]
        with patch('datasets.shd_speaker_split.load_manifest',
                   return_value=dict(validation_speaker_ids=[0, 3], role='development')):
            for layer, key, value in [('l1', 'cell', 'se_adlif'),
                                      ('l2', 'rho_slow', .95),
                                      ('l1', 'theta_max', 2.)]:
                invalid = copy.deepcopy(dth)
                invalid[layer][key] = value
                with self.subTest(layer=layer, key=key), self.assertRaisesRegex(ValueError, 'neuron'):
                    validate_config(invalid)

    def test_dth_objective_factory_state_rng_forward_and_gradients(self):
        ordinary = OmegaConf.create(composed('DTH_SE_adLIF'))
        smoothed = OmegaConf.create(composed('DTH_LS_SE_adLIF'))
        for cfg in (ordinary, smoothed):
            cfg.l1.input_size = 4
            cfg.l1.n_neurons = 5
            cfg.l2.input_size = 5
            cfg.l2.n_neurons = 5
            cfg.l_out.input_size = 5
            cfg.l_out.n_neurons = 3
            cfg.dataset.num_classes = 3
        devices = ['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device), contextlib.redirect_stdout(io.StringIO()):
                torch.manual_seed(123)
                a = build_training_model(ordinary)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(123)
                b = build_training_model(smoothed)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertIsInstance(b.l1, DTHSEAdLIF)
                self.assertIsInstance(b.l2, DTHSEAdLIF)
                self.assertEqual(set(a.state_dict()), set(b.state_dict()))
                for name, value in a.state_dict().items():
                    torch.testing.assert_close(value, b.state_dict()[name], rtol=0, atol=0)
                a.to(device).eval()
                b.to(device).eval()
                x = torch.randn(2, 20, 4, device=device)*4
                target = torch.tensor([[-1, 0], [-1, 2]], dtype=torch.int32, device=device)
                blocks = torch.ones(2, 20, dtype=torch.long, device=device)
                blocks[:, :2] = 0
                with torch.no_grad():
                    plain = a(x)
                    prediction = b(x)
                torch.testing.assert_close(plain, prediction, rtol=0, atol=0)
                _, aloss, _ = a.process_predictions_and_compute_losses(plain, target, blocks)
                _, bloss, _ = b.process_predictions_and_compute_losses(prediction, target, blocks)
                torch.testing.assert_close(aloss, bloss, rtol=0, atol=0)
                b.train()
                reduced, loss, _ = b.process_predictions_and_compute_losses(b(x), target, blocks)
                valid = target.flatten().ne(-1)
                expected = F.cross_entropy(reduced[valid].float(),
                    target.flatten()[valid].long(), label_smoothing=.1)
                torch.testing.assert_close(loss, expected)
                loss.backward()
                for layer in (b.l1, b.l2):
                    for parameter in (layer.threshold_logits, layer.homeostatic_gain,
                                      layer.transient_gain, layer.weight):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_source_identity_includes_actual_dth_implementation(self):
        name = 'models/ours/dth_adlif.py'
        expected = hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        self.assertEqual(forward_identity('SHD')[name], expected)


if __name__ == '__main__':
    unittest.main()
