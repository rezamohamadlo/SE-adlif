import unittest
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

import runner
from measure_shd_costs import ScalarOperations, batch_scalar_operations, operation_profiles
from models.ours.amt_adlif import AMTSEAdLIF
from models.ours.mt_adlif import MTSEAdLIF
from tests.test_measure_shd_costs import tiny_model


def config(**kwargs):
    return OmegaConf.create(dict(input_size=4, n_neurons=6,
                                 tau_u_range=[5, 25], tau_w_range=[60, 300],
                                 q=120, **kwargs))


class AMTTests(unittest.TestCase):
    def test_zero_gain_recovers_mt_on_cpu_and_cuda_if_available(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                torch.manual_seed(42)
                mt = MTSEAdLIF(config()).to(device)
                amt = AMTSEAdLIF(config()).to(device)
                missing = amt.load_state_dict(mt.state_dict(), strict=False)
                self.assertEqual(missing.missing_keys, ['mix_activity_gain'])
                self.assertEqual(missing.unexpected_keys, [])
                a, b = mt.initial_state(2, device), amt.initial_state(2, device)
                for _ in range(25):
                    x = torch.randn(2, 4, device=device) * 20
                    za, a = mt(x, a)
                    zb, b = amt(x, b)
                    torch.testing.assert_close(za, zb, rtol=0, atol=0)
                    for old, new in zip(a, b[:4]):
                        torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_gate_is_bounded_signed_and_uses_previous_sample_local_spikes(self):
        cell = AMTSEAdLIF(config())
        with torch.no_grad():
            cell.mix_activity_gain.fill_(2)
        state = list(cell.initial_state(2))
        with torch.no_grad():
            state[1][0].fill_(1)
        _, new = cell(torch.zeros(2, 4), tuple(state))
        torch.testing.assert_close(new[4][0], torch.full((6,), 0.1))
        torch.testing.assert_close(new[5][0], torch.full((6,), 0.01))
        torch.testing.assert_close(new[4][1], torch.zeros(6))
        mix = cell.adaptation_mix(new[4], new[5])
        self.assertTrue((mix[0] > 0.5).all())
        torch.testing.assert_close(mix[1], torch.full((6,), 0.5))
        self.assertTrue(((mix > 0) & (mix < 1)).all())

    def test_scan_batch_independence_and_finite_nonzero_gate_gradient(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                torch.manual_seed(123)
                cell = AMTSEAdLIF(config()).to(device)
                with torch.no_grad():
                    cell.mix_activity_gain.fill_(1)
                x = torch.randn(2, 35, 4, device=device) * 30
                state, spikes = cell.initial_state(2, device), []
                for frame in x.unbind(1):
                    z, state = cell(frame, state)
                    spikes.append(z)
                result = cell.layer_forward(x)
                torch.testing.assert_close(result, torch.stack(spikes, dim=1))
                alone = cell.layer_forward(x[:1])
                torch.testing.assert_close(result[:1], alone)
                # Perturbing the future must leave the prefix unchanged.
                changed = x.clone()
                changed[:, 20:] += 100
                torch.testing.assert_close(result[:, :20], cell.layer_forward(changed)[:, :20])
                result.sum().backward()
                for name in ['mix_logits', 'mix_activity_gain', 'tau_w_slow_trainer.weight']:
                    grad = dict(cell.named_parameters())[name].grad
                    self.assertIsNotNone(grad, name)
                    self.assertTrue(torch.isfinite(grad).all(), name)
                self.assertGreater(cell.mix_activity_gain.grad.abs().sum().item(), 0)
                states, outputs = cell.layer_forward_with_states(x)
                torch.testing.assert_close(outputs, result.detach())
                self.assertEqual(states.shape[0], 6)

    def test_cost_counter_supports_new_model(self):
        with torch.inference_mode():
            model = tiny_model('amt_se_adlif')
            profiles = operation_profiles(model)
            inputs = torch.randn(3, 7, 4)
            with ScalarOperations() as count:
                model(inputs)
            self.assertEqual(count.counts, batch_scalar_operations(profiles, 3, 7))

    def test_invalid_trace_rates(self):
        for fast, slow in [(-0.1, 0.9), (0.99, 0.9), (0.9, 1.0)]:
            with self.assertRaises(ValueError):
                AMTSEAdLIF(config(mix_rho_fast=fast, mix_rho_slow=slow))

    def test_three_dataset_three_seed_matched_baseline_and_new_paths(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'results') as temporary:
            output = Path(temporary).relative_to(runner.ROOT).as_posix()
            with patch.object(runner, 'RUN_LOGDIR', output):
                plan = runner.build_plan('AMT_SE_adLIF')
                mt = runner.build_plan('MT_SE_adLIF')
        self.assertEqual(len(plan), 9)
        self.assertEqual({p['seed'] for p in plan}, {42, 123, 456})
        for new, old in zip(plan, mt):
            self.assertIn('AMT_SE_adLIF', new['directory'].parts)
            self.assertEqual(new['status'], 'new')
            for key in ['dataset', 'lr', 'batch_size', 'n_epochs', 'early_stopping',
                        'dropout', 'loss_agg', 'paper_baseline_version']:
                self.assertEqual(new['config'][key], old['config'][key], key)
            for name in ['l1', 'l2']:
                expected = dict(old['config'][name], cell='amt_se_adlif',
                                mix_rho_fast=0.9, mix_rho_slow=0.99)
                self.assertEqual(new['config'][name], expected)

    def test_configs_portable_to_other_checkout_and_working_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / 'checkout'
            shutil.copytree(runner.ROOT / 'config', root / 'config')
            hydra_cwd = Path(temporary) / 'hydra_job'
            hydra_cwd.mkdir()
            try:
                os.chdir(hydra_cwd)
                with patch.object(runner, 'ROOT', root), patch.object(
                        runner, 'RUN_LOGDIR', 'results/experiments/amt_portability_test'):
                    plan = runner.build_plan('AMT_SE_adLIF')
            finally:
                os.chdir(original_cwd)
        self.assertEqual(len(plan), 9)
        for item in plan:
            self.assertTrue(item['directory'].is_relative_to(root))
            self.assertTrue(item['config']['logdir'].startswith('results/experiments/amt_portability_test/'))


if __name__ == '__main__':
    unittest.main()
