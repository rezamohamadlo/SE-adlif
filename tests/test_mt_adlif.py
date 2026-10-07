import unittest

import torch
from omegaconf import OmegaConf

from models.alif import SEAdLIF
from models.ours.mt_adlif import MTSEAdLIF


class MTTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = OmegaConf.create(dict(input_size=4, n_neurons=6,
            tau_u_range=[5, 25], tau_w_range=[60, 300], q=120))
        self.model = MTSEAdLIF(self.cfg)

    def test_identical_memories_recover_reference(self):
        original = SEAdLIF(self.cfg)
        original.load_state_dict({k: v for k, v in self.model.state_dict().items()
                                 if k in original.state_dict()})
        self.model.tau_w_slow_trainer.load_state_dict(original.tau_w_trainer.state_dict())
        x = torch.randn(2, 30, 4) * 5
        torch.testing.assert_close(self.model.layer_forward(x), original.layer_forward(x))

    def test_gradients_and_constraints(self):
        self.model.layer_forward(torch.randn(2, 30, 4) * 5).sum().backward()
        for parameter in (self.model.mix_logits, self.model.tau_w_trainer.weight,
                          self.model.tau_w_slow_trainer.weight, self.model.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        with torch.no_grad():
            self.model.tau_w_slow_trainer.weight.fill_(2)
        self.model.apply_parameter_constraints()
        tau = self.model.tau_w_slow_trainer.get_tau()
        self.assertTrue(((tau >= 300) & (tau <= 600)).all())

    def test_step_scan_and_sample_isolation(self):
        x = torch.randn(2, 30, 4) * 5
        state = self.model.initial_state(2)
        outputs = []
        for cur in x.unbind(1):
            z, state = self.model(cur, state)
            outputs.append(z)
        expected = torch.stack(outputs, 1)
        torch.testing.assert_close(self.model.layer_forward(x), expected)
        torch.testing.assert_close(self.model.layer_forward(x[:1]), expected[:1])
        states, actual = self.model.layer_forward_with_states(x)
        self.assertEqual(states.shape, (4, 2, 31, 6))
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
