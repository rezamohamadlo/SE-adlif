"""Multi-timescale adaptation currents with DTH activity-controlled thresholds."""
import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.ours.mt_adlif import MTSEAdLIF
from models.ours.dth_adlif import DTHSEAdLIF
from models.helpers import spike_grad_injection_function


class MTDTHSEAdLIF(MTSEAdLIF):
    """State: membrane, spikes, fast/slow currents, fast/slow activity traces.

    Both mechanisms retain their original update ordering. Activity traces use
    previous spikes; adaptation currents use the post-reset membrane/current spikes.
    """
    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.rho_fast = float(cfg.get('rho_fast', 0.9))
        self.rho_slow = float(cfg.get('rho_slow', 0.99))
        self.target_rate = float(cfg.get('target_rate', 0.05))
        self.theta_min = float(cfg.get('theta_min', 0.5))
        self.theta_max = float(cfg.get('theta_max', 1.5))
        if not 0 <= self.rho_fast < self.rho_slow < 1:
            raise ValueError('Require 0 <= rho_fast < rho_slow < 1')
        if not self.theta_min < 1 < self.theta_max or not 0 <= self.target_rate <= 1:
            raise ValueError('Invalid threshold bounds or target rate')
        probability = (1 - self.theta_min) / (self.theta_max - self.theta_min)
        self.threshold_logits = Parameter(self.weight.new_full(
            (self.out_features,), math.log(probability / (1 - probability))))
        self.homeostatic_gain = Parameter(self.weight.new_zeros(()))
        self.transient_gain = Parameter(self.weight.new_zeros(()))

    dynamic_threshold = DTHSEAdLIF.dynamic_threshold

    def initial_state(self, batch_size, device=None):
        state = super().initial_state(batch_size, device)
        rate = torch.full_like(state[1], self.target_rate)
        return (*state, rate, rate.clone())

    def step(self, recurrent, alpha, beta_fast, thr, a, b, u_rest, carry, cur):
        u_prev, z_prev, fast_prev, slow_prev, rate_fast, rate_slow = carry
        activity = z_prev.mean(-1, keepdim=True)
        rate_fast = self.rho_fast * rate_fast + (1 - self.rho_fast) * activity
        rate_slow = self.rho_slow * rate_slow + (1 - self.rho_slow) * activity
        threshold = self.dynamic_threshold(rate_fast, rate_slow)
        beta_slow = self.tau_w_slow_trainer.get_decay()
        mix = self.mix_logits.sigmoid()
        if self.use_recurrent:
            cur = cur + F.linear(z_prev, recurrent)
        adaptation = mix * fast_prev + (1 - mix) * slow_prev
        u = alpha * u_prev + (1 - alpha) * (cur - adaptation)
        z = spike_grad_injection_function(u - threshold, self.alpha, self.c)
        u = u * (1 - z.detach()) + u_rest * z.detach()
        drive = (a * u + b * z) * self.q
        fast = beta_fast * fast_prev + (1 - beta_fast) * drive
        slow = beta_slow * slow_prev + (1 - beta_slow) * drive
        return (u, z, fast, slow, rate_fast, rate_slow), z

    def apply_parameter_constraints(self):
        super().apply_parameter_constraints()
        with torch.no_grad():
            self.homeostatic_gain.clamp_(0, 5)
            self.transient_gain.clamp_(0, 5)
