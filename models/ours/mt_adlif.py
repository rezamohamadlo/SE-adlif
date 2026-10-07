"""SE-adLIF with fast and slow adaptation currents."""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.alif import SEAdLIF
from models.helpers import generic_scan, generic_scan_with_states, spike_grad_injection_function
from module.tau_trainers import get_tau_trainer_class


class MTSEAdLIF(SEAdLIF):
    """Two adaptation memories with a convex, neuron-specific learned mixture.

    State order is (membrane, spikes, fast adaptation, slow adaptation).
    Both currents use the post-reset membrane and current spikes, preserving
    the symplectic update order. The two branches share a, b and q.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        # Remove the original instance closure to expose our four-state step.
        del self.step
        slow_range = cfg.get("tau_w_slow_range", [300, 600])
        if len(slow_range) != 2 or not 0 < slow_range[0] <= slow_range[1]:
            raise ValueError("tau_w_slow_range must be positive and ordered")
        self.tau_w_slow_trainer = get_tau_trainer_class(self.train_tau_w_method)(
            self.out_features, self.dt, slow_range[0], slow_range[1],
            device=device, dtype=dtype,
        )
        self.tau_w_slow_trainer.reset_parameters()
        self.mix_logits = Parameter(self.weight.new_zeros(self.out_features))

    def initial_state(self, batch_size, device=None):
        u, z, fast = super().initial_state(batch_size, device)
        return u, z, fast, torch.zeros_like(fast)

    def step(self, recurrent, alpha, beta_fast, thr, a, b, u_rest, carry, cur):
        u_prev, z_prev, fast_prev, slow_prev = carry
        beta_slow = self.tau_w_slow_trainer.get_decay()
        mix = self.mix_logits.sigmoid()
        if self.use_recurrent:
            cur = cur + F.linear(z_prev, recurrent)
        adaptation = mix * fast_prev + (1 - mix) * slow_prev
        u = alpha * u_prev + (1 - alpha) * (cur - adaptation)
        z = spike_grad_injection_function(u - thr, self.alpha, self.c)
        u = u * (1 - z.detach()) + u_rest * z.detach()
        drive = (a * u + b * z) * self.q
        fast = beta_fast * fast_prev + (1 - beta_fast) * drive
        slow = beta_slow * slow_prev + (1 - beta_slow) * drive
        return (u, z, fast, slow), z

    def forward(self, input_tensor, states):
        current = F.linear(input_tensor, self.weight, self.bias)
        rest = self.u0 if self.use_u_rest else torch.zeros_like(self.u0)
        state, z = self.step(
            self.recurrent, self.tau_u_trainer.get_decay(),
            self.tau_w_trainer.get_decay(), self.thr, self.a, self.b,
            rest, states, current,
        )
        return z, state

    def _scan(self, inputs, with_states=False):
        current = F.linear(inputs, self.weight, self.bias)
        alpha = self.tau_u_trainer.get_decay()
        beta = self.tau_w_trainer.get_decay()
        rest = self.u0 if self.use_u_rest else torch.zeros_like(self.u0)

        def step(carry, cur):
            return self.step(self.recurrent, alpha, beta, self.thr,
                                 self.a, self.b, rest, carry, cur)

        scan = generic_scan_with_states if with_states else generic_scan
        return scan(step, self.initial_state(inputs.shape[0], inputs.device),
                    current, self.unroll)

    def layer_forward(self, inputs):
        return self._scan(inputs)[..., :self.num_out_neuron]

    @torch.no_grad()
    def layer_forward_with_states(self, inputs):
        states, outputs = self._scan(inputs, with_states=True)
        return states[..., :self.num_out_neuron], outputs[..., :self.num_out_neuron]

    def apply_parameter_constraints(self):
        super().apply_parameter_constraints()
        self.tau_w_slow_trainer.apply_parameter_constraints()
