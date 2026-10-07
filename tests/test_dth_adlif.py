import unittest

import torch
from omegaconf import OmegaConf

from models.alif import SEAdLIF
from models.ours.dth_adlif import DTHSEAdLIF


class DTHTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = OmegaConf.create(dict(input_size=4, n_neurons=6,
            tau_u_range=[5, 25], tau_w_range=[60, 300], q=120))
        self.model = DTHSEAdLIF(self.cfg)

    def test_zero_gains_match_reference(self):
        original = SEAdLIF(self.cfg)
        original.load_state_dict({k: v for k, v in self.model.state_dict().items()
                                 if k in original.state_dict()})
        x = torch.randn(2, 20, 4)
        torch.testing.assert_close(self.model.layer_forward(x), original.layer_forward(x))

    def test_gradients_and_bounds(self):
        x = torch.randn(2, 20, 4) * 5
        self.model.layer_forward(x).sum().backward()
        for parameter in (self.model.threshold_logits, self.model.homeostatic_gain,
                          self.model.transient_gain, self.model.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.threshold_logits.grad.abs().sum().item(), 0)
        threshold = self.model.dynamic_threshold(torch.rand(2, 6), torch.rand(2, 6))
        self.assertTrue((threshold >= self.model.theta_min).all())
        self.assertTrue((threshold <= self.model.theta_max).all())

    def test_step_scan_and_sample_isolation(self):
        with torch.no_grad():
            self.model.homeostatic_gain.fill_(1)
            self.model.transient_gain.fill_(1)
        x = torch.randn(2, 20, 4) * 5
        state = self.model.initial_state(2)
        output = []
        for cur in x.unbind(1):
            z, state = self.model(cur, state)
            output.append(z)
        expected = torch.stack(output, 1)
        torch.testing.assert_close(self.model.layer_forward(x), expected)
        torch.testing.assert_close(self.model.layer_forward(x[:1]), expected[:1])
        states, actual = self.model.layer_forward_with_states(x)
        self.assertEqual(states.shape, (5, 2, 21, 6))
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
