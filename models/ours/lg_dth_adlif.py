"""Local–global dual-timescale threshold extension of SE-adLIF."""

import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.ours.dth_adlif import DTHSEAdLIF
from models.helpers import spike_grad_injection_function


class LGDTHSEAdLIF(DTHSEAdLIF):
    """Previous spikes drive sample-local traces and centered corrections.

    Effective coefficients are bound * tanh(raw), initially zero. Disabled
    corrections retain their checkpoint parameters but do not enter the graph.
    State is passed explicitly and recreated for every sequence by _scan.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.local_slow_enabled = bool(cfg.get("local_slow_enabled", True))
        self.local_transient_enabled = bool(cfg.get("local_transient_enabled", True))
        self.local_coefficient_bound = float(cfg.get("local_coefficient_bound", 5.0))
        if not math.isfinite(self.local_coefficient_bound) or self.local_coefficient_bound <= 0:
            raise ValueError("local_coefficient_bound must be finite and positive")
        self.lambda_raw = Parameter(self.weight.new_zeros(self.out_features))
        self.mu_raw = Parameter(self.weight.new_zeros(self.out_features))

        def step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
            u_prev, z_prev, w_prev, fast_prev, slow_prev = carry
            fast = self.rho_fast * fast_prev + (1 - self.rho_fast) * z_prev
            slow = self.rho_slow * slow_prev + (1 - self.rho_slow) * z_prev
            threshold = self.dynamic_threshold(fast, slow)
            if self.use_recurrent:
                cur = cur + F.linear(z_prev, recurrent)
            u = alpha * u_prev + (1 - alpha) * (cur - w_prev)
            z = spike_grad_injection_function(u - threshold, self.alpha, self.c)
            u = u * (1 - z.detach()) + u_rest * z.detach()
            w = beta * w_prev + (1 - beta) * (a * u + b * z) * self.q
            return (u, z, w, fast, slow), z

        self.step = step_fn

    @property
    def lambda_coefficient(self):
        return self.local_coefficient_bound * self.lambda_raw.tanh()

    @property
    def mu_coefficient(self):
        return self.local_coefficient_bound * self.mu_raw.tanh()

    def dynamic_threshold(self, fast, slow):
        global_fast = fast.mean(dim=-1, keepdim=True)
        global_slow = slow.mean(dim=-1, keepdim=True)
        logits = (self.threshold_logits
                  + self.homeostatic_gain * (global_slow - self.target_rate)
                  - self.transient_gain * (global_fast - global_slow).abs())
        if self.local_slow_enabled:
            logits = logits + self.lambda_coefficient * (slow - global_slow)
        if self.local_transient_enabled:
            logits = logits + self.mu_coefficient * (
                (fast - slow) - (global_fast - global_slow))
        return self.theta_min + (self.theta_max - self.theta_min) * logits.sigmoid()
