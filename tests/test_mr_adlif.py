import unittest

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.alif import SEAdLIF
from models.ours.mr_adlif import MRSEAdLIF


def config(k=1):
    return OmegaConf.create(dict(input_size=4, n_neurons=6,
        tau_u_range=[5, 25], tau_w_range=[60, 300], q=120,
        adaptation_update_interval=k))


class MRTests(unittest.TestCase):
    def test_k1_equivalence(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.float32, torch.float64):
                torch.manual_seed(42)
                reference = SEAdLIF(config()).to(device=device, dtype=dtype)
                model = MRSEAdLIF(config()).to(device=device, dtype=dtype)
                model.load_state_dict(reference.state_dict(), strict=True)
                inputs = torch.randn(2, 24, 4, device=device, dtype=dtype) * 20
                ref_state = tuple(v.to(dtype=dtype) for v in reference.initial_state(2, device))
                state = model.initial_state(2, device)
                errors = dict(membrane=0., spikes=0., u=0., w=0.)
                reference_spikes = []
                for x in inputs.unbind(1):
                    pre = []
                    for cell, carry in ((reference, ref_state), (model, state)):
                        current = F.linear(x, cell.weight, cell.bias) + F.linear(carry[1], cell.recurrent)
                        alpha = cell.tau_u_trainer.get_decay()
                        pre.append(alpha * carry[0] + (1-alpha) * (current-carry[2]))
                    z_ref, ref_state = reference(x, ref_state)
                    reference_spikes.append(z_ref)
                    z, state = model(x, state)
                    for name, left, right in [('membrane', *pre), ('spikes', z_ref, z),
                                              ('u', ref_state[0], state[0]),
                                              ('w', ref_state[2], state[2])]:
                        errors[name] = max(errors[name], (left-right).abs().max().item())
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                torch.testing.assert_close(model.layer_forward(inputs), torch.stack(reference_spikes, dim=1))
                # The original scan initializes float32 state, even with float64 weights.
                if dtype == torch.float32:
                    torch.testing.assert_close(model.layer_forward(inputs), reference.layer_forward(inputs))
                print(f'K=1 {device} {dtype}: maximum errors {errors}')

    def test_k4_boundaries_and_block_equation(self):
        torch.manual_seed(42)
        model = MRSEAdLIF(config(4))
        state = model.initial_state(2)
        total_u = torch.zeros_like(state[2])
        total_s = torch.zeros_like(state[2])
        for t in range(1, 11):
            previous_w = state[2].clone()
            z, state = model(torch.randn(2, 4) * 20, state)
            total_u = total_u + state[0]
            total_s = total_s + z
            if t % 4 == 0:
                beta_k = model.tau_w_trainer.get_decay() ** 4
                expected = beta_k * previous_w + (1-beta_k) * (
                    model.a * (total_u / 4) + model.b * (total_s / 4)) * model.q
                torch.testing.assert_close(state[2], expected, rtol=0, atol=0)
                self.assertGreater((state[2]-previous_w).abs().max().item(), 0)
                self.assertEqual(state[3].abs().sum().item(), 0)
                self.assertEqual(state[4].abs().sum().item(), 0)
                total_u, total_s = torch.zeros_like(total_u), torch.zeros_like(total_s)
            else:
                torch.testing.assert_close(state[2], previous_w, rtol=0, atol=0)
                torch.testing.assert_close(state[3], total_u)
                torch.testing.assert_close(state[4], total_s)
            self.assertEqual(state[5], t % 4)
        self.assertEqual(model.initial_state(2)[5], 0)

    def test_k4_gradients_scan_and_dtype(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            torch.manual_seed(42)
            model = MRSEAdLIF(config(4)).to(device=device, dtype=torch.float64)
            inputs = torch.randn(2, 17, 4, device=device, dtype=torch.float64) * 20
            output = model.layer_forward(inputs)
            output.sum().backward()
            for name in ('weight', 'tau_u_trainer.weight', 'tau_w_trainer.weight', 'a', 'b'):
                parameter = dict(model.named_parameters())[name]
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
            states, spikes = model.layer_forward_with_states(inputs)
            self.assertEqual(states.shape, (6, 2, 18, 6))
            torch.testing.assert_close(output, spikes)
            torch.testing.assert_close(model.layer_forward(inputs[:1]), output[:1])

    def test_invalid_intervals(self):
        for k in (0, -1, 1.5, True, '4', None):
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                MRSEAdLIF(config(k))


if __name__ == '__main__':
    unittest.main()
