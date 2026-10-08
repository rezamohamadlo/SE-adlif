import contextlib
import io
import math
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from models.alif import SEAdLIF
from models.ours.fp_adlif import FPSEAdLIF
from models.pl_module import MLPSNN, layer_map


def config(**overrides):
    values = dict(input_size=4, n_neurons=6, tau_u_range=[5, 25],
                  tau_w_range=[60, 300], q=120, dt=1.0,
                  frequency_range_hz=[1., 50.])
    values.update(overrides)
    return OmegaConf.create(values)


class FPTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_formula_units_and_subthreshold_eigenvalues(self):
        for dt in (0.5, 1., 2.):
            model = FPSEAdLIF(config(dt=dt), dtype=torch.float64)
            alpha, beta = model.tau_u_trainer.get_decay(), model.tau_w_trainer.get_decay()
            phi = 2 * math.pi * model.frequency_hz * dt * 1e-3
            physical_a = model.derived_a(alpha, beta) * model.q
            expected = (alpha + beta - 2 * (alpha*beta).sqrt()*phi.cos()) / ((1-alpha)*(1-beta))
            torch.testing.assert_close(physical_a, expected, rtol=1e-10, atol=1e-10)
            # Autonomous spike-free SE update of [u,w], excluding recurrence.
            drive = (1-beta)*physical_a
            matrix = torch.stack((torch.stack((alpha, -(1-alpha)), -1),
                                  torch.stack((drive*alpha, beta-drive*(1-alpha)), -1)), -2)
            eigenvalues = torch.linalg.eigvals(matrix)
            torch.testing.assert_close(eigenvalues.abs(), (alpha*beta).sqrt()[:, None].expand(-1, 2))
            recovered_hz = eigenvalues.angle().abs() / (2*math.pi*dt*1e-3)
            torch.testing.assert_close(recovered_hz, model.frequency_hz[:, None].expand(-1, 2))
            self.assertTrue((eigenvalues.abs() < 1).all())

    def test_original_step_reset_and_surrogate_are_preserved(self):
        model = FPSEAdLIF(config())
        reference = SEAdLIF(config())
        reference.load_state_dict({**reference.state_dict(), **{
            k: v for k, v in model.state_dict().items() if k in reference.state_dict()}})
        with torch.no_grad():
            reference.a.copy_(model.derived_a())
        state, ref_state = model.initial_state(2), reference.initial_state(2)
        for x in (torch.randn(2, 4)*20 for _ in range(24)):
            z, state = model(x, state)
            ref_z, ref_state = reference(x, ref_state)
            torch.testing.assert_close(z, ref_z, rtol=0, atol=0)
            for actual, expected in zip(state, ref_state):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Same inherited dynamics also give the same local surrogate derivative.
        x = torch.randn(2, 4, requires_grad=True)*20
        z, _ = model(x, model.initial_state(2))
        ref_z, _ = reference(x, reference.initial_state(2))
        torch.testing.assert_close(torch.autograd.grad(z.sum(), x)[0],
                                   torch.autograd.grad(ref_z.sum(), x)[0], rtol=0, atol=0)

    def test_backward_scan_dtype_device_and_sequence_reset(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for dtype in (torch.float32, torch.float64):
                model = FPSEAdLIF(config(), device=device, dtype=dtype).to(device)
                inputs = torch.randn(2, 30, 4, device=device, dtype=dtype)*20
                output = model.layer_forward(inputs)
                output.sum().backward()
                for name in ('weight', 'recurrent', 'frequency_logits',
                             'tau_u_trainer.weight', 'tau_w_trainer.weight', 'b'):
                    grad = dict(model.named_parameters())[name].grad
                    self.assertIsNotNone(grad, name)
                    self.assertTrue(torch.isfinite(grad).all(), name)
                    self.assertGreater(grad.abs().sum().item(), 0, name)
                state, spikes = model.initial_state(2, device), []
                for x in inputs.unbind(1):
                    z, state = model(x, state)
                    spikes.append(z)
                torch.testing.assert_close(output, torch.stack(spikes, 1))
                torch.testing.assert_close(model.layer_forward(inputs[:1]), output[:1])
                histories, actual = model.layer_forward_with_states(inputs)
                self.assertEqual(histories.shape, (3, 2, 31, 6))
                torch.testing.assert_close(actual, output)
                self.assertEqual(output.dtype, dtype)
                self.assertNotIn('a', dict(model.named_parameters()))

    def test_bounds_constraints_and_reset(self):
        model = FPSEAdLIF(config())
        with torch.no_grad():
            model.frequency_logits.copy_(torch.linspace(-100, 100, 6))
            model.tau_u_trainer.weight.fill_(2)
            model.tau_w_trainer.weight.fill_(-2)
            model.b.fill_(3)
        model.apply_parameter_constraints()
        self.assertTrue(((model.frequency_hz >= 1) & (model.frequency_hz <= 50)).all())
        self.assertTrue(torch.isfinite(model.derived_a()).all())
        self.assertTrue((model.b <= 2).all())
        model.reset_parameters()
        self.assertTrue(torch.isfinite(model.derived_a()).all())

    def test_long_subthreshold_stability(self):
        model = FPSEAdLIF(config(thr=1e6, use_recurrent=False), dtype=torch.float64)
        with torch.no_grad():
            model.frequency_logits.copy_(torch.linspace(-20, 20, 6))
            state = (torch.ones(2, 6, dtype=torch.float64),
                     torch.zeros(2, 6, dtype=torch.float64),
                     torch.zeros(2, 6, dtype=torch.float64))
            for _ in range(2000):
                _, state = model(torch.zeros(2, 4, dtype=torch.float64), state)
            self.assertTrue(all(torch.isfinite(value).all() for value in state))
            self.assertLess(state[0].abs().max().item(), 1e-8)
            self.assertLess(state[2].abs().max().item(), 1e-8)

    def test_invalid_configuration(self):
        for bounds in ([0, 50], [50, 1], [1, 500], [1, float('nan')], [1]):
            with self.assertRaisesRegex(ValueError, 'frequency_range_hz'):
                FPSEAdLIF(config(frequency_range_hz=bounds))
        for values in (dict(dt=0), dict(q=0), dict(tau_u_range=[0, 5])):
            with self.assertRaises(ValueError):
                FPSEAdLIF(config(**values))

    def test_hydra_registry_and_full_model_smoke(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'config'), version_base=None):
            cfg = compose(config_name='main', overrides=[
                'experiment=SHD_FP_SE_adLIF', 'frequency_range_hz=[2.0,40.0]',
                'batch_size=512', 'dataset.batch_size=512'])
        self.assertIs(layer_map[cfg.l1.cell], FPSEAdLIF)
        self.assertEqual(cfg.l1.n_neurons, 360)
        self.assertEqual(cfg.l2.n_neurons, 360)
        self.assertEqual(cfg.loss_agg, 'sum_softmax_over_time')
        self.assertEqual(cfg.lr, .01)
        self.assertEqual(cfg.n_epochs, 300)
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            with contextlib.redirect_stdout(io.StringIO()):
                model = MLPSNN(cfg).to(device)
            output = model(torch.randn(2, 16, 140, device=device)*5)
            self.assertEqual(output.shape, (2, 16, 20))
            torch.nn.functional.cross_entropy(output.mean(1), torch.tensor([0, 1], device=device)).backward()
            for cell in (model.l1, model.l2):
                self.assertIsInstance(cell, FPSEAdLIF)
                self.assertTrue(torch.isfinite(cell.frequency_logits.grad).all())


if __name__ == '__main__':
    unittest.main()
