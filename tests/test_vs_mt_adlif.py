"""Non-training equation, reconstruction and cost checks for voltage-sensitive MT."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch

import measure_run_costs
from measure_shd_costs import ScalarOperations, operation_profiles, batch_scalar_operations
from models.ours.mt_adlif import MTSEAdLIF
from models.ours.vs_mt_adlif import VSMTSEAdLIF
from models.pl_module import MLPSNN
from tests.test_measure_shd_costs import tiny_model
from runner_vs_mt_ssc import parameter_record


def config(**extra):
    return OmegaConf.create(dict(input_size=4, n_neurons=6, tau_u_range=[5, 25],
                                 tau_w_range=[60, 300], q=120, **extra))


class VoltageSensitivityTests(unittest.TestCase):
    def devices(self):
        return ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

    def test_initial_equivalence_rng_storage_and_gradients_cpu_cuda(self):
        for device in self.devices():
            with self.subTest(device=device):
                torch.manual_seed(42)
                mt = MTSEAdLIF(config()).to(device)
                rng = torch.get_rng_state()
                torch.manual_seed(42)
                vs = VSMTSEAdLIF(config()).to(device)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                for name, value in mt.state_dict().items():
                    torch.testing.assert_close(value, vs.state_dict()[name], rtol=0, atol=0)
                self.assertNotEqual(vs.a.data_ptr(), vs.a_slow.data_ptr())
                self.assertFalse(hasattr(vs, 'b_slow'))
                x = torch.randn(2, 31, 4, device=device) * 8
                expected, actual = mt.layer_forward(x), vs.layer_forward(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected.sum().backward()
                actual.sum().backward()
                for grad in (vs.a.grad, vs.a_slow.grad):
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(), 0)
                torch.testing.assert_close(mt.a.grad, vs.a.grad + vs.a_slow.grad,
                                           rtol=2e-4, atol=1e-5)

    def test_only_slow_voltage_drive_changes(self):
        cell = VSMTSEAdLIF(config())
        base = MTSEAdLIF(config())
        base.load_state_dict({k: v for k, v in cell.state_dict().items() if k != 'a_slow'})
        with torch.no_grad():
            cell.a_slow.copy_(cell.a + .2)
            for model in (base, cell):
                model.thr.fill_(100)  # Keep post-reset voltage nonzero without spiking.
        x = torch.ones(2, 4)
        z, state = cell(x, cell.initial_state(2))
        expected_z, original = base(x, base.initial_state(2))
        torch.testing.assert_close(z, expected_z)
        for a, b in zip(state[:3], original[:3]):
            torch.testing.assert_close(a, b)
        change = (1-cell.tau_w_slow_trainer.get_decay()) * .2 * state[0] * cell.q
        torch.testing.assert_close(state[3]-original[3], change, rtol=1e-4, atol=1e-6)

    def test_constraints_reset_dtype_and_sequence_cpu_cuda(self):
        for device in self.devices():
            with self.subTest(device=device):
                cell = VSMTSEAdLIF(config(a_range=[.2, .8]), device=device, dtype=torch.float64)
                self.assertEqual(cell.a_slow.dtype, torch.float64)
                self.assertEqual(cell.a_slow.device.type, device)
                # The inherited MT state initializer uses float32. Match the
                # supported runner path, which transfers the complete module.
                cell = cell.float().to(device)
                with torch.no_grad():
                    cell.a.fill_(-1)
                    cell.a_slow[:3].fill_(-1)
                    cell.a_slow[3:].fill_(2)
                cell.apply_parameter_constraints()
                self.assertTrue(((cell.a_slow >= .2) & (cell.a_slow <= .8)).all())
                self.assertTrue(((cell.a >= .2) & (cell.a <= .8)).all())
                cell.reset_parameters()
                torch.testing.assert_close(cell.a, cell.a_slow, rtol=0, atol=0)
                with torch.no_grad():
                    cell.a_slow.mul_(.5)
                x = torch.randn(2, 17, 4, device=device) * 8
                state, spikes = cell.initial_state(2, device), []
                for frame in x.unbind(1):
                    z, state = cell(frame, state)
                    spikes.append(z)
                actual = cell.layer_forward(x)
                torch.testing.assert_close(actual, torch.stack(spikes, 1))
                states, outputs = cell.layer_forward_with_states(x)
                self.assertEqual(states.shape[0], 4)
                torch.testing.assert_close(outputs, actual)
                torch.testing.assert_close(actual[:1], cell.layer_forward(x[:1]))
                changed = x.clone()
                changed[:, 9:] += 100
                torch.testing.assert_close(actual[:, :9], cell.layer_forward(changed)[:, :9])

    def test_checkpoint_reconstruction_parameters_and_costs_cpu_cuda(self):
        for device in self.devices():
            with self.subTest(device=device), tempfile.TemporaryDirectory() as tmp:
                mt, model = tiny_model('mt_se_adlif'), tiny_model('vs_mt_se_adlif')
                count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
                self.assertEqual(count(model)-count(mt), 12)
                with torch.no_grad():
                    model.l1.a_slow.mul_(.6)
                path = Path(tmp)/'best.ckpt'
                torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(cfg=model.hparams.cfg)), path)
                checkpoint = torch.load(path, weights_only=False, map_location='cpu')
                with contextlib.redirect_stdout(io.StringIO()):
                    restored = MLPSNN(checkpoint['hyper_parameters']['cfg']).eval().to(device)
                restored.load_state_dict(checkpoint['state_dict'], strict=True)
                x = torch.randn(3, 7, 4, device=device)
                model.to(device)
                with torch.inference_mode():
                    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
                    profiles = operation_profiles(model)
                    with ScalarOperations() as direct:
                        model(x)
                    self.assertEqual(direct.counts, batch_scalar_operations(profiles, 3, 7))
                record = parameter_record(restored)
                self.assertEqual(len(record['l1']['a_slow']), 6)
                self.assertEqual(len(record['l2']['tau_slow']), 6)

    def test_ssc_cost_evaluator_without_training(self):
        with tempfile.TemporaryDirectory() as tmp, torch.inference_mode():
            directory = Path(tmp)
            model = tiny_model('vs_mt_se_adlif')
            cfg = model.hparams.cfg
            cfg.random_seed, cfg.device = 42, 'cpu'
            cfg.dataset._target_ = 'datasets.ssc.SSCLDM'
            cfg.loss_agg = 'mean_membrane_potentials'
            path = directory/'best.ckpt'
            torch.save(dict(state_dict=model.state_dict(), hyper_parameters=dict(cfg=cfg), epoch=2), path)
            (directory/'completed.txt').touch()
            (directory/'TRAINING_SUMMARY.md').write_text('# Synthetic inference-only fixture\n')
            x = torch.randn(3, 4, 4)
            block = torch.ones(3, 4, dtype=torch.long)
            prediction = model(x).mean(1).argmax(-1)
            targets = torch.stack([torch.full_like(prediction, -1), prediction], 1)
            batch = (x, targets, block, torch.full((3,), 4))
            with patch.object(measure_run_costs.shared, 'selected_checkpoint', return_value=path), \
                 patch.object(measure_run_costs, 'prepare_batches', return_value=([batch], 'synthetic')), \
                 patch.object(measure_run_costs.shared, 'saved_test_accuracy', return_value=1.), \
                 contextlib.redirect_stdout(io.StringIO()):
                record = measure_run_costs.collect(directory, 'SSC')
            self.assertEqual(record['status'], 'complete')
            self.assertEqual(record['measurements']['reproduced_test_acc'], 1.)
            self.assertTrue((directory/'COST_RESULTS.json').exists())
