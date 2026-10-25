import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import runner
from measure_shd_costs import ScalarOperations, batch_scalar_operations, operation_profiles
from models.ours.dth_adlif import DTHSEAdLIF
from models.ours.lt_dth_adlif import LTDTHSEAdLIF
from models.pl_module import MLPSNN
from tests.test_measure_shd_costs import tiny_model


def config(**kwargs):
    return OmegaConf.create(dict(input_size=4, n_neurons=6, tau_u_range=[5, 25],
                                 tau_w_range=[60, 300], q=120, **kwargs))


class LearnedTraceDTHTests(unittest.TestCase):
    def test_initial_rates_reproduce_dth_dynamics_cpu_cuda(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                torch.manual_seed(42)
                baseline = DTHSEAdLIF(config()).to(device)
                learned = LTDTHSEAdLIF(config()).to(device)
                missing = learned.load_state_dict(baseline.state_dict(), strict=False)
                self.assertEqual(set(missing.missing_keys), {'trace_fast_logit', 'trace_slow_logit'})
                self.assertEqual(missing.unexpected_keys, [])
                fast, slow = learned.trace_decays()
                self.assertAlmostEqual(fast.item(), .9, places=6)
                self.assertAlmostEqual(slow.item(), .99, places=6)
                # Activate threshold feedback to check the traces themselves,
                # not just the zero-gain reduction to the reference neuron.
                for cell in (baseline, learned):
                    with torch.no_grad():
                        cell.homeostatic_gain.fill_(1)
                        cell.transient_gain.fill_(.4)
                a, b = baseline.initial_state(2, device), learned.initial_state(2, device)
                for _ in range(35):
                    x = torch.randn(2, 4, device=device) * 20
                    za, a = baseline(x, a)
                    zb, b = learned(x, b)
                    torch.testing.assert_close(za, zb, rtol=0, atol=0)
                    for old, new in zip(a, b):
                        torch.testing.assert_close(old, new, rtol=1e-5, atol=1e-6)

    def test_ordering_remains_strict_at_extreme_logits(self):
        cell = LTDTHSEAdLIF(config())
        for a in (-1000, 0, 1000):
            for b in (-1000, 0, 1000):
                with torch.no_grad():
                    cell.trace_fast_logit.fill_(a)
                    cell.trace_slow_logit.fill_(b)
                fast, slow = cell.trace_decays()
                self.assertTrue(0 < fast.item() < slow.item() < 1)

    def test_scan_causality_sample_independence_and_gradients(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                torch.manual_seed(123)
                cell = LTDTHSEAdLIF(config()).to(device)
                with torch.no_grad():
                    cell.homeostatic_gain.fill_(1)
                    cell.transient_gain.fill_(.4)
                x = torch.randn(2, 37, 4, device=device) * 20
                state, spikes = cell.initial_state(2, device), []
                for frame in x.unbind(1):
                    z, state = cell(frame, state)
                    spikes.append(z)
                actual = cell.layer_forward(x)
                torch.testing.assert_close(actual, torch.stack(spikes, 1))
                torch.testing.assert_close(actual[:1], cell.layer_forward(x[:1]))
                changed = x.clone()
                changed[:, 20:] += 100
                torch.testing.assert_close(actual[:, :20], cell.layer_forward(changed)[:, :20])
                actual.sum().backward()
                for name in ['trace_fast_logit', 'trace_slow_logit']:
                    grad = getattr(cell, name).grad
                    self.assertIsNotNone(grad, name)
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    self.assertGreater(grad.abs().sum().item(), 0, name)
                states, result = cell.layer_forward_with_states(x)
                self.assertEqual(states.shape[0], 5)
                torch.testing.assert_close(result, actual.detach())

    def test_cost_profile_and_four_extra_trainable_parameters(self):
        with torch.inference_mode():
            baseline = tiny_model('dth_se_adlif')
            model = tiny_model('lt_dth_se_adlif')
            count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.assertEqual(count_parameters(model) - count_parameters(baseline), 4)
            profiles = operation_profiles(model)
            inputs = torch.randn(3, 7, 4)
            with ScalarOperations() as count:
                model(inputs)
            self.assertEqual(count.counts, batch_scalar_operations(profiles, 3, 7))

    def test_invalid_ranges_and_initial_rates(self):
        for kwargs in [dict(trace_rho_fast_range=[0, .98]),
                       dict(trace_rho_fast_range=[.5, 1]),
                       dict(trace_rho_slow_range=[.97, .999]),
                       dict(trace_rho_fast_range=[.91, .98]),
                       dict(trace_rho_slow_range=[.991, .9995])]:
            with self.assertRaises(ValueError):
                LTDTHSEAdLIF(config(**kwargs))

    def test_three_dataset_three_seed_fair_plan_and_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'results') as temporary:
            output = Path(temporary).relative_to(runner.ROOT).as_posix()
            with patch.object(runner, 'RUN_LOGDIR', output):
                plan = runner.build_plan('LT_DTH_SE_adLIF')
                baseline = runner.build_plan('DTH_SE_adLIF')
        self.assertEqual(len(plan), 9)
        for item, original in zip(plan, baseline):
            self.assertTrue(item['config']['run_test'])
            self.assertEqual(item['status'], 'new')
            self.assertIn('LT_DTH_SE_adLIF', item['directory'].parts)
            for key in ['dataset', 'lr', 'n_epochs', 'batch_size', 'early_stopping', 'dropout',
                        'loss_agg', 'paper_baseline_version', 'evaluation_protocol']:
                self.assertEqual(item['config'][key], original['config'][key], key)
            for layer in ['l1', 'l2']:
                expected = dict(original['config'][layer], cell='lt_dth_se_adlif',
                                trace_rho_fast_range=[.5, .98], trace_rho_slow_range=[.985, .9995])
                self.assertEqual(item['config'][layer], expected)
        # Trusted local state dictionaries must preserve learned rates exactly.
        model = tiny_model('lt_dth_se_adlif')
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'state.pt'
            torch.save(model.state_dict(), checkpoint)
            with contextlib.redirect_stdout(io.StringIO()):
                restored = MLPSNN(model.hparams.cfg).eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        inputs = torch.randn(2, 13, 4)
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)

    def test_portable_configuration_from_another_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copytree(runner.ROOT / 'config', root / 'config')
            with patch.object(runner, 'ROOT', root), patch.object(
                    runner, 'RUN_LOGDIR', 'results/experiments/lt_dth_portability_test'):
                plan = runner.build_plan('LT_DTH_SE_adLIF')
        self.assertEqual(len(plan), 9)
        for item in plan:
            self.assertTrue(item['config']['logdir'].startswith('results/experiments/lt_dth_portability_test/'))


if __name__ == '__main__':
    unittest.main()
