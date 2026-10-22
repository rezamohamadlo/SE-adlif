"""Activity-dependent mixing of MT's fast and slow adaptation memories."""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.helpers import spike_grad_injection_function
from models.ours.mt_adlif import MTSEAdLIF


class AMTSEAdLIF(MTSEAdLIF):
    """Causal layer activity modulates each neuron's adaptation mixture.

    State: membrane, spikes, fast/slow adaptation, fast/slow activity traces.
    Activity is sample-local and uses previous spikes only. Zero gate gains
    recover MT exactly when its parameters are copied; fresh training does
    not load an MT checkpoint. Threshold and adaptation drive stay unchanged.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.mix_rho_fast = float(cfg.get("mix_rho_fast", 0.9))
        self.mix_rho_slow = float(cfg.get("mix_rho_slow", 0.99))
        if not 0 <= self.mix_rho_fast < self.mix_rho_slow < 1:
            raise ValueError("Require 0 <= mix_rho_fast < mix_rho_slow < 1")
        self.mix_activity_gain = Parameter(self.weight.new_zeros(self.out_features))

    def initial_state(self, batch_size, device=None):
        state = super().initial_state(batch_size, device)
        # Full neuron-shaped traces preserve the generic state-scan interface.
        trace = torch.zeros_like(state[1])
        return (*state, trace, trace.clone())

    def adaptation_mix(self, rate_fast, rate_slow):
        return (self.mix_logits
                + self.mix_activity_gain * (rate_fast - rate_slow)).sigmoid()

    def step(self, recurrent, alpha, beta_fast, thr, a, b, u_rest, carry, cur):
        u_prev, z_prev, fast_prev, slow_prev, rate_fast, rate_slow = carry
        activity = z_prev.mean(dim=-1, keepdim=True)
        rate_fast = self.mix_rho_fast * rate_fast + (1 - self.mix_rho_fast) * activity
        rate_slow = self.mix_rho_slow * rate_slow + (1 - self.mix_rho_slow) * activity
        mix = self.adaptation_mix(rate_fast, rate_slow)
        beta_slow = self.tau_w_slow_trainer.get_decay()
        if self.use_recurrent:
            cur = cur + F.linear(z_prev, recurrent)
        adaptation = mix * fast_prev + (1 - mix) * slow_prev
        u = alpha * u_prev + (1 - alpha) * (cur - adaptation)
        z = spike_grad_injection_function(u - thr, self.alpha, self.c)
        u = u * (1 - z.detach()) + u_rest * z.detach()
        drive = (a * u + b * z) * self.q
        fast = beta_fast * fast_prev + (1 - beta_fast) * drive
        slow = beta_slow * slow_prev + (1 - beta_slow) * drive
        return (u, z, fast, slow, rate_fast, rate_slow), z
