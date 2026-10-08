"""Fixed multi-rate adaptation with the SE-adLIF update ordering."""

import torch
import torch.nn.functional as F

from models.alif import SEAdLIF
from models.helpers import generic_scan, spike_grad_injection_function


class MRSEAdLIF(SEAdLIF):
    """State: (u, z, w, sum_u, sum_s, phase).

    Phase is a sequence-local Python integer, shared across the layer/batch.
    This avoids GPU synchronization and avoids computing w on held steps.
    An incomplete final block remains accumulated; no partial-block flush occurs.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        interval = cfg.get("adaptation_update_interval", 1)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("adaptation_update_interval must be a positive integer")
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.adaptation_update_interval = interval
        # The parent assigns an instance closure; expose our multi-rate step.
        del self.step

    def initial_state(self, batch_size, device=None):
        # Follow parameter dtype, including float64 equivalence checks.
        target_device = self.u0.device if device is None else device
        zeros = torch.zeros(batch_size, self.out_features,
                            device=target_device, dtype=self.u0.dtype)
        return (self.u0.unsqueeze(0), zeros.clone(), zeros.clone(),
                zeros.clone(), zeros.clone(), 0)

    def step(self, recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
        u_prev, z_prev, w_prev, sum_u, sum_s, phase = carry
        if self.use_recurrent:
            cur = cur + F.linear(z_prev, recurrent)
        u = alpha * u_prev + (1.0 - alpha) * (cur - w_prev)
        z = spike_grad_injection_function(u - thr, self.alpha, self.c)
        u = u * (1 - z.detach()) + u_rest * z.detach()
        sum_u = sum_u + u
        sum_s = sum_s + z
        phase += 1
        w = w_prev
        if phase == self.adaptation_update_interval:
            k = self.adaptation_update_interval
            beta_k = beta if k == 1 else beta ** k
            w = beta_k * w_prev + (1.0 - beta_k) * (
                a * (sum_u / k) + b * (sum_s / k)
            ) * self.q
            sum_u = torch.zeros_like(sum_u)
            sum_s = torch.zeros_like(sum_s)
            phase = 0
        return (u, z, w, sum_u, sum_s, phase), z

    def _sequence_step(self):
        alpha = self.tau_u_trainer.get_decay()
        beta = self.tau_w_trainer.get_decay()
        rest = self.u0 if self.use_u_rest else torch.zeros_like(self.u0)

        def step(carry, cur):
            return self.step(self.recurrent, alpha, beta, self.thr,
                             self.a, self.b, rest, carry, cur)
        return step

    def layer_forward(self, inputs):
        current = F.linear(inputs, self.weight, self.bias)
        outputs = generic_scan(self._sequence_step(),
                               self.initial_state(inputs.shape[0], inputs.device),
                               current, self.unroll)
        return outputs[..., :self.num_out_neuron]

    @torch.no_grad()
    def layer_forward_with_states(self, inputs):
        current = F.linear(inputs, self.weight, self.bias)
        state = self.initial_state(inputs.shape[0], inputs.device)
        step = self._sequence_step()

        def snapshot(carry):
            tensors = [value.expand(inputs.shape[0], -1) for value in carry[:5]]
            return torch.stack(tensors + [torch.full_like(tensors[0], carry[5])])

        histories, outputs = [snapshot(state)], []
        for cur in current.unbind(1):
            state, z = step(state, cur)
            histories.append(snapshot(state))
            outputs.append(z)
        return (torch.stack(histories, dim=2)[..., :self.num_out_neuron],
                torch.stack(outputs, dim=1)[..., :self.num_out_neuron])
