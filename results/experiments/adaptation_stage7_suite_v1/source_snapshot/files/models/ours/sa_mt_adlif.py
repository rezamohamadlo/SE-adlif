"""MT with separately learned fast and slow spike-adaptation strengths."""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.helpers import spike_grad_injection_function
from models.ours.mt_adlif import MTSEAdLIF


class SAMTSEAdLIF(MTSEAdLIF):
    """Share membrane coupling a, but learn b_slow separately from fast b.

    The slow coefficient starts as a copy of b, preserving MT's initial
    dynamics without loading a trained checkpoint. Both spike strengths use
    the inherited b_range; q, time constants, mixture, reset and state order
    remain MT's. No extra recurrent state is introduced.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.b_slow = Parameter(self.b.detach().clone())

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'b_slow'):
            with torch.no_grad():
                self.b_slow.copy_(self.b)

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
        fast_drive = (a * u + b * z) * self.q
        slow_drive = (a * u + self.b_slow * z) * self.q
        fast = beta_fast * fast_prev + (1 - beta_fast) * fast_drive
        slow = beta_slow * slow_prev + (1 - beta_slow) * slow_drive
        return (u, z, fast, slow), z

    def apply_parameter_constraints(self):
        super().apply_parameter_constraints()
        with torch.no_grad():
            self.b_slow.clamp_(min=self.b_range[0], max=self.b_range[1])
