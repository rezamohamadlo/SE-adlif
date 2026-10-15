import contextlib
import io
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

import runner
from models.ours.dth_adlif import DTHSEAdLIF
from models.ours.lg_dth_adlif import LGDTHSEAdLIF
from models.pl_module import MLPSNN
from run import TrustedLocalCheckpointIO
from pytorch_lightning.strategies import SingleDeviceStrategy


def config(**kwargs):
    values = dict(input_size=4, n_neurons=6, tau_u_range=[5, 25],
                  tau_w_range=[60, 300], q=120)
    values.update(kwargs)
    return OmegaConf.create(values)


class LGDTHTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_exact_initialization_and_disabled_equivalence(self):
        for disabled in (False, True):
            model = LGDTHSEAdLIF(config(local_slow_enabled=not disabled,
                                       local_transient_enabled=not disabled))
            reference = DTHSEAdLIF(config())
            reference.load_state_dict({k: v for k, v in model.state_dict().items()
                                       if k in reference.state_dict()})
            if disabled:
                with torch.no_grad():
                    model.lambda_raw.fill_(2)
                    model.mu_raw.fill_(-3)
            x = torch.randn(2, 50, 4) * 20
            actual_states, actual = model.layer_forward_with_states(x)
            expected_states, expected = reference.layer_forward_with_states(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_states[:3], expected_states[:3], rtol=0, atol=0)

    def test_causal_updates_formula_and_isolation(self):
        model = LGDTHSEAdLIF(config())
        with torch.no_grad():
            model.homeostatic_gain.fill_(1.2)
            model.transient_gain.fill_(.8)
            model.lambda_raw.copy_(torch.linspace(-1, 1, 6))
            model.mu_raw.copy_(torch.linspace(1, -1, 6))
        x = torch.randn(2, 31, 4) * 20
        state = model.initial_state(2)
        outputs = []
        for cur in x.unbind(1):
            previous = state
            z, state = model(cur, state)
            outputs.append(z)
            fast = model.rho_fast * previous[3] + (1-model.rho_fast)*previous[1]
            slow = model.rho_slow * previous[4] + (1-model.rho_slow)*previous[1]
            torch.testing.assert_close(state[3], fast, rtol=0, atol=0)
            torch.testing.assert_close(state[4], slow, rtol=0, atol=0)
            gf, gs = fast.mean(-1, keepdim=True), slow.mean(-1, keepdim=True)
            logits = (model.threshold_logits + 1.2*(gs-model.target_rate)
                      - .8*(gf-gs).abs() + model.lambda_coefficient*(slow-gs)
                      + model.mu_coefficient*((fast-slow)-(gf-gs)))
            torch.testing.assert_close(model.dynamic_threshold(fast, slow),
                model.theta_min+(model.theta_max-model.theta_min)*logits.sigmoid())
        output = torch.stack(outputs, 1)
        torch.testing.assert_close(model.layer_forward(x), output, rtol=0, atol=0)
        torch.testing.assert_close(model.layer_forward(x[:1]), output[:1])
        changed = x.clone()
        changed[1] *= -100
        changed[:, 15:] *= -100
        torch.testing.assert_close(model.layer_forward(changed)[:1, :15], output[:1, :15])
        # Other sequences cannot mutate either initial traces or later scans.
        model.layer_forward(changed)
        torch.testing.assert_close(model.layer_forward(x), output, rtol=0, atol=0)
        fresh = model.initial_state(2)
        torch.testing.assert_close(fresh[3], torch.full_like(fresh[3], model.target_rate))
        self.assertNotEqual(fresh[3].data_ptr(), fresh[4].data_ptr())

    def test_backward_shapes_and_device(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            model = LGDTHSEAdLIF(config()).to(device)
            x = torch.randn(2, 30, 4, device=device)*20
            y = model.layer_forward(x)
            self.assertEqual(y.shape, (2, 30, 6))
            y.sum().backward()
            for p in (model.lambda_raw, model.mu_raw, model.weight):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(), 0)
            states, actual = model.layer_forward_with_states(x)
            self.assertEqual(states.shape, (5, 2, 31, 6))
            torch.testing.assert_close(actual, y)

    def test_bounds_stability_and_ablation_gradients(self):
        for slow, transient in [(True, False), (False, True), (True, True), (False, False)]:
            model = LGDTHSEAdLIF(config(local_slow_enabled=slow,
                                       local_transient_enabled=transient))
            model.layer_forward(torch.randn(2, 30, 4)*20).sum().backward()
            for p, enabled in [(model.lambda_raw, slow), (model.mu_raw, transient)]:
                if enabled:
                    self.assertGreater(p.grad.abs().sum().item(), 0)
                else:
                    self.assertIsNone(p.grad)
            with torch.no_grad():
                model.lambda_raw.copy_(torch.linspace(-1e20, 1e20, 6))
                model.mu_raw.copy_(torch.linspace(1e20, -1e20, 6))
                model.threshold_logits.copy_(torch.linspace(-1e20, 1e20, 6))
            threshold = model.dynamic_threshold(torch.rand(2, 6), torch.rand(2, 6))
            self.assertTrue(torch.isfinite(threshold).all())
            self.assertTrue((threshold >= model.theta_min).all())
            self.assertTrue((threshold <= model.theta_max).all())
            self.assertTrue((model.lambda_coefficient.abs() <= 5).all())
            self.assertTrue((model.mu_coefficient.abs() <= 5).all())
        for bound in [0, -1, float('inf'), float('nan')]:
            with self.assertRaises(ValueError):
                LGDTHSEAdLIF(config(local_coefficient_bound=bound))

    def test_protocol_registry_counts_and_resume_plan(self):
        reference = runner.build_plan('DTH_SE_adLIF', seeds=(42,), datasets=('SHD',))[0]
        for variant in ['LG_DTH_SE_adLIF', 'LG_DTH_SE_adLIF_slow', 'LG_DTH_SE_adLIF_transient']:
            plan = runner.build_plan(variant, seeds=(42, 123, 456), datasets=('SHD',))
            cfg = OmegaConf.create(plan[0]['config'])
            self.assertEqual(cfg.l1.n_neurons, 360)
            self.assertEqual(cfg.l2.n_neurons, 360)
            for key in reference['config']:
                if key not in ['l1', 'l2', 'exp_name', 'model_variant', 'logdir', 'hydra']:
                    self.assertEqual(plan[0]['config'][key], reference['config'][key], key)
            with contextlib.redirect_stdout(io.StringIO()):
                model = MLPSNN(cfg)
                baseline = MLPSNN(OmegaConf.create(reference['config']))
            count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
            self.assertEqual(count(model)-count(baseline), 1440)
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'ROOT', Path(directory)):
            shutil.copytree(Path(__file__).resolve().parents[1]/'config', Path(directory)/'config')
            item = runner.build_plan('LG_DTH_SE_adLIF', seeds=(42,), datasets=('SHD',))[0]
            item['directory'].mkdir(parents=True)
            import json
            (item['directory']/'FULL_RUN_CONFIG.json').write_text(json.dumps(item['config']))
            (item['directory']/'ckpt').mkdir()
            (item['directory']/'ckpt/last.ckpt').touch()
            resumed = runner.build_plan('LG_DTH_SE_adLIF', seeds=(42,), datasets=('SHD',))[0]
            self.assertEqual(resumed['status'], 'resume')

    def test_lightning_smoke_checkpoint_and_optimizer_resume(self):
        cfg = OmegaConf.create(runner.build_plan('LG_DTH_SE_adLIF', seeds=(42,), datasets=('SHD',))[0]['config'])
        # Synthetic short sequences; retain full 360+360 architecture and loss.
        cfg.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        inputs = torch.randn(2, 16, 140)*20
        targets = torch.tensor([[0], [1]])
        blocks = torch.zeros(2, 16, dtype=torch.long)
        loader = DataLoader(TensorDataset(inputs, targets, blocks), batch_size=2)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(cfg)
            kwargs = dict(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                          devices=1, logger=False, enable_checkpointing=False,
                          enable_progress_bar=False, enable_model_summary=False,
                          limit_train_batches=1, limit_val_batches=1, num_sanity_val_steps=0,
                          gradient_clip_val=cfg.gradient_clip_val, default_root_dir=directory)
            kwargs['strategy'] = SingleDeviceStrategy(
                device=torch.device(cfg.device), checkpoint_io=TrustedLocalCheckpointIO())
            trainer = pl.Trainer(max_epochs=1, **kwargs)
            trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
            path = str(Path(directory)/'smoke.ckpt')
            trainer.save_checkpoint(path)
            saved = TrustedLocalCheckpointIO().load_checkpoint(path, map_location='cpu')
            restored = MLPSNN(saved['hyper_parameters']['cfg'])
            restored.load_state_dict(saved['state_dict'], strict=True)
            model.cpu().eval()
            restored.eval()
            torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
            kwargs['strategy'] = SingleDeviceStrategy(
                device=torch.device(cfg.device), checkpoint_io=TrustedLocalCheckpointIO())
            resumed = pl.Trainer(max_epochs=2, **kwargs)
            resumed.fit(restored, train_dataloaders=loader, val_dataloaders=loader, ckpt_path=path)
            self.assertEqual(resumed.global_step, 2)
            self.assertTrue(resumed.optimizers[0].state)


if __name__ == '__main__':
    unittest.main()
