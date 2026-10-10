import contextlib
import io
import unittest
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from models.alif import SEAdLIF
from models.ours.da_adlif import DASEAdLIF
from models.pl_module import MLPSNN, layer_map


def config(**overrides):
    values = dict(input_size=4, n_neurons=6, tau_u_range=[5, 25],
                  tau_w_range=[60, 300], q=120, dt=1.0, delay_frames=[1, 2, 4])
    values.update(overrides)
    return OmegaConf.create(values)


class DATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_zero_gate_preserves_original_dynamics(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for dtype in (torch.float32, torch.float64):
                reference = SEAdLIF(config()).to(device=device, dtype=dtype)
                model = DASEAdLIF(config()).to(device=device, dtype=dtype)
                model.load_state_dict(reference.state_dict(), strict=False)
                self.assertEqual(model.delay_gate.abs().sum().item(), 0)
                inputs = torch.randn(2, 24, 4, device=device, dtype=dtype)*20
                state = model.initial_state(2, device)
                ref_state = tuple(v.to(dtype) for v in reference.initial_state(2, device))
                expected_spikes = []
                for x in inputs.unbind(1):
                    spikes, state = model(x, state)
                    ref_spikes, ref_state = reference(x, ref_state)
                    expected_spikes.append(ref_spikes)
                    torch.testing.assert_close(spikes, ref_spikes, rtol=0, atol=0)
                    for actual, expected in zip(state[:3], ref_state):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected = torch.stack(expected_spikes, 1)
                torch.testing.assert_close(model.layer_forward(inputs), expected, rtol=0, atol=0)
                if dtype == torch.float32:
                    torch.testing.assert_close(model.layer_forward(inputs),
                                               reference.layer_forward(inputs), rtol=0, atol=0)

    def test_causal_delay_average_and_batch_isolation(self):
        model = DASEAdLIF(config(), dtype=torch.float64)
        with torch.no_grad():
            model.delay_gate.fill_(1)
            model.delay_logits.zero_()
        currents = torch.arange(1, 1+2*7*6, dtype=torch.float64).reshape(2, 7, 6)
        history = model.initial_state(2)[3]
        self.assertEqual(history.shape, (2, 4, 6))
        for t, current in enumerate(currents.unbind(1)):
            actual, history = model.filter_current(current, history)
            expected = sum((currents[:, t-d] if t >= d else torch.zeros_like(current))
                           for d in (1, 2, 4))/3
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(history[:, 0], current)
            for offset in range(1, min(t+1, 4)):
                torch.testing.assert_close(history[:, offset], currents[:, t-offset])
        self.assertEqual(model.initial_state(2)[3].abs().sum().item(), 0)
        # Per-neuron logits and gate must implement a convex mixture.
        with torch.no_grad():
            model.delay_gate.copy_(torch.linspace(0, 1, 6))
            model.delay_logits.copy_(torch.arange(18).reshape(6, 3)/5)
        actual, _ = model.filter_current(currents[:, -1], history)
        delayed = history[:, [0, 1, 3]]
        mixed = (delayed * model.delay_logits.softmax(-1).T.unsqueeze(0)).sum(1)
        expected = (1-model.delay_gate)*currents[:, -1] + model.delay_gate*mixed
        torch.testing.assert_close(actual, expected)

    def test_history_retains_gradients(self):
        model = DASEAdLIF(config())
        with torch.no_grad():
            model.delay_gate.fill_(1)
        history = model.initial_state(2)[3]
        previous = torch.randn(2, 6, requires_grad=True)
        _, history = model.filter_current(previous, history)
        actual, _ = model.filter_current(torch.zeros_like(previous), history)
        actual.sum().backward()
        torch.testing.assert_close(previous.grad, torch.full_like(previous, 1/3))

    def test_scan_backward_dtype_device_and_state_reset(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for dtype in (torch.float32, torch.float64):
                for gate in (0., .4):
                    torch.manual_seed(42)
                    model = DASEAdLIF(config()).to(device=device, dtype=dtype)
                    with torch.no_grad():
                        model.delay_gate.fill_(gate)
                    inputs = torch.randn(2, 30, 4, device=device, dtype=dtype)*20
                    output = model.layer_forward(inputs)
                    self.assertEqual(output.dtype, dtype)
                    self.assertTrue(torch.isfinite(output).all())
                    output.sum().backward()
                    names = ['weight', 'recurrent', 'tau_u_trainer.weight',
                             'tau_w_trainer.weight', 'a', 'b', 'delay_gate']
                    if gate > 0:
                        names.append('delay_logits')
                    for name in names:
                        grad = dict(model.named_parameters())[name].grad
                        self.assertIsNotNone(grad, name)
                        self.assertTrue(torch.isfinite(grad).all(), name)
                        self.assertGreater(grad.abs().sum().item(), 0, name)
                    torch.testing.assert_close(model.layer_forward(inputs[:1]), output[:1])
                    histories, actual = model.layer_forward_with_states(inputs)
                    self.assertEqual(len(histories), 4)
                    for state in histories[:3]:
                        self.assertEqual(state.shape, (2, 31, 6))
                        self.assertTrue(torch.isfinite(state).all())
                    self.assertEqual(histories[3].shape, (2, 31, 4, 6))
                    self.assertTrue(torch.isfinite(histories[3]).all())
                    torch.testing.assert_close(actual, output)

    def test_constraints_reset_and_invalid_delays(self):
        model = DASEAdLIF(config())
        with torch.no_grad():
            model.delay_gate.copy_(torch.linspace(-1, 2, 6))
        model.apply_parameter_constraints()
        self.assertTrue(((model.delay_gate >= 0) & (model.delay_gate <= 1)).all())
        model.reset_parameters()
        self.assertEqual(model.delay_gate.abs().sum().item(), 0)
        self.assertEqual(model.delay_logits.abs().sum().item(), 0)
        for delays in ([], [0], [-1], [1, 1], [1.5], [True], ['1'], None):
            with self.subTest(delays=delays), self.assertRaises(ValueError):
                DASEAdLIF(config(delay_frames=delays))

    def test_hydra_registry_and_model_construction(self):
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'config'), version_base=None):
            cfg = compose(config_name='main', overrides=['experiment=SHD_DA_SE_adLIF'])
        self.assertIs(layer_map[cfg.l1.cell], DASEAdLIF)
        self.assertEqual(cfg.l1.n_neurons, 360)
        self.assertEqual(cfg.l2.n_neurons, 360)
        self.assertEqual(cfg.loss_agg, 'sum_softmax_over_time')
        self.assertEqual(cfg.lr, .01)
        self.assertEqual(cfg.n_epochs, 300)
        with contextlib.redirect_stdout(io.StringIO()):
            model = MLPSNN(cfg)
        output = model(torch.randn(2, 16, 140)*5)
        self.assertEqual(output.shape, (2, 16, 20))
        torch.nn.functional.cross_entropy(output.mean(1), torch.tensor([0, 1])).backward()
        for cell in (model.l1, model.l2):
            self.assertIsInstance(cell, DASEAdLIF)
            self.assertTrue(torch.isfinite(cell.delay_gate.grad).all())


if __name__ == '__main__':
    unittest.main()
