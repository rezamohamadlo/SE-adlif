import contextlib
import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import runner
from measure_shd_costs import ScalarOperations, batch_scalar_operations, operation_profiles
from models.ours.mt_adlif import MTSEAdLIF
from models.ours.sa_mt_adlif import SAMTSEAdLIF
from models.pl_module import MLPSNN
from tests.test_measure_shd_costs import tiny_model


def config(**kwargs):
    return OmegaConf.create(dict(input_size=4, n_neurons=6, tau_u_range=[5, 25],
                                 tau_w_range=[60, 300], q=120, **kwargs))


class SeparateAdaptationMTTests(unittest.TestCase):
    def test_fresh_initialization_matches_mt_and_tied_gradients_cpu_cuda(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                torch.manual_seed(42)
                mt = MTSEAdLIF(config()).to(device)
                torch.manual_seed(42)
                separate = SAMTSEAdLIF(config()).to(device)
                for key, value in mt.state_dict().items():
                    torch.testing.assert_close(value, separate.state_dict()[key], rtol=0, atol=0)
                torch.testing.assert_close(separate.b_slow, mt.b, rtol=0, atol=0)
                self.assertNotEqual(separate.b.data_ptr(), separate.b_slow.data_ptr())
                x = torch.randn(2, 35, 4, device=device) * 20
                expected = mt.layer_forward(x)
                actual = separate.layer_forward(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.sum().backward()
                actual.sum().backward()
                self.assertTrue(torch.isfinite(separate.b_slow.grad).all())
                self.assertGreater(separate.b_slow.grad.abs().sum().item(), 0)
                torch.testing.assert_close(mt.b.grad, separate.b.grad + separate.b_slow.grad,
                                           rtol=1e-4, atol=1e-5)

    def test_distinct_strength_changes_only_slow_drive_on_first_step(self):
        cell = SAMTSEAdLIF(config())
        baseline = MTSEAdLIF(config())
        baseline.load_state_dict({k: v for k, v in cell.state_dict().items()
                                 if k != 'b_slow'}, strict=True)
        for model in (cell, baseline):
            with torch.no_grad():
                model.b.fill_(.5)
                model.thr.fill_(-1)  # Force a spike in this synthetic equation check.
        with torch.no_grad():
            cell.b_slow.fill_(1)
        x = torch.zeros(2, 4)
        z, state = cell(x, cell.initial_state(2))
        expected_z, expected = baseline(x, baseline.initial_state(2))
        torch.testing.assert_close(z, expected_z)
        for a, b in zip(state[:3], expected[:3]):
            torch.testing.assert_close(a, b)
        difference = (1 - cell.tau_w_slow_trainer.get_decay()) * .5 * cell.q
        torch.testing.assert_close(state[3] - expected[3], difference.expand_as(state[3]))

    def test_constraints_and_reset_keep_same_coefficient_range(self):
        cell = SAMTSEAdLIF(config(b_range=[.2, .8]))
        with torch.no_grad():
            cell.b_slow[:3].fill_(-1)
            cell.b_slow[3:].fill_(2)
        cell.apply_parameter_constraints()
        self.assertTrue((cell.b_slow >= .2).all())
        self.assertTrue((cell.b_slow <= .8).all())
        cell.reset_parameters()
        torch.testing.assert_close(cell.b_slow, cell.b, rtol=0, atol=0)

    def test_scan_causality_and_sample_independence(self):
        cell = SAMTSEAdLIF(config())
        with torch.no_grad():
            cell.b_slow.mul_(.5)
        x = torch.randn(2, 23, 4) * 20
        state, spikes = cell.initial_state(2), []
        for frame in x.unbind(1):
            z, state = cell(frame, state)
            spikes.append(z)
        result = cell.layer_forward(x)
        torch.testing.assert_close(result, torch.stack(spikes, 1))
        torch.testing.assert_close(result[:1], cell.layer_forward(x[:1]))
        changed = x.clone()
        changed[:, 13:] += 100
        torch.testing.assert_close(result[:, :13], cell.layer_forward(changed)[:, :13])
        states, outputs = cell.layer_forward_with_states(x)
        self.assertEqual(states.shape[0], 4)
        torch.testing.assert_close(outputs, result)

    def test_cost_profile_parameter_count_and_checkpoint_roundtrip(self):
        baseline = tiny_model('mt_se_adlif')
        model = tiny_model('sa_mt_se_adlif')
        count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
        self.assertEqual(count(model) - count(baseline), 12)  # Two layers of six neurons.
        inputs = torch.randn(3, 7, 4)
        with torch.inference_mode():
            profiles = operation_profiles(model)
            with ScalarOperations() as measured:
                model(inputs)
            self.assertEqual(measured.counts, batch_scalar_operations(profiles, 3, 7))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'state.pt'
            torch.save(model.state_dict(), path)
            with contextlib.redirect_stdout(io.StringIO()):
                restored = MLPSNN(model.hparams.cfg).eval()
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
        torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)

    def test_fair_nine_job_plan(self):
        with tempfile.TemporaryDirectory(dir=runner.ROOT / 'results') as temporary:
            output = Path(temporary).relative_to(runner.ROOT).as_posix()
            with patch.object(runner, 'RUN_LOGDIR', output):
                plan = runner.build_plan('SA_MT_SE_adLIF')
                baseline = runner.build_plan('MT_SE_adLIF')
        self.assertEqual(len(plan), 9)
        for item, original in zip(plan, baseline):
            self.assertTrue(item['config']['run_test'])
            self.assertEqual(item['status'], 'new')
            self.assertIn('SA_MT_SE_adLIF', item['directory'].parts)
            for key in ['dataset', 'lr', 'batch_size', 'n_epochs', 'early_stopping', 'dropout',
                        'loss_agg', 'paper_baseline_version', 'evaluation_protocol']:
                self.assertEqual(item['config'][key], original['config'][key], key)
            for layer in ['l1', 'l2']:
                self.assertEqual(item['config'][layer], dict(original['config'][layer], cell='sa_mt_se_adlif'))

    def test_portability_from_other_checkout_and_hydra_working_directory(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'checkout'
            shutil.copytree(runner.ROOT / 'config', root / 'config')
            hydra_cwd = Path(temporary) / 'hydra_job'
            hydra_cwd.mkdir()
            try:
                os.chdir(hydra_cwd)
                with patch.object(runner, 'ROOT', root), patch.object(
                        runner, 'RUN_LOGDIR', 'results/experiments/sa_mt_v1'):
                    plan = runner.build_plan('SA_MT_SE_adLIF')
            finally:
                os.chdir(original_cwd)
        self.assertEqual(len(plan), 9)
        for item in plan:
            self.assertTrue(item['config']['logdir'].startswith('results/experiments/sa_mt_v1/'))


if __name__ == '__main__':
    unittest.main()
