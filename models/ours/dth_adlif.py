"""Experimental dual-timescale threshold extension of SE-adLIF."""

import math

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.alif import SEAdLIF
from models.helpers import generic_scan, generic_scan_with_states, spike_grad_injection_function


class DTHSEAdLIF(SEAdLIF):
    """Causal, per-sample layer activity controls bounded neuron thresholds.

    The original symplectic membrane/adaptation update is retained. Fast and
    slow traces use previous spikes; no information is shared between samples.
    Zero-initialized gains and theta=1 reproduce the original initial dynamics.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.rho_fast = float(cfg.get("rho_fast", 0.9))
        self.rho_slow = float(cfg.get("rho_slow", 0.99))
        self.target_rate = float(cfg.get("target_rate", 0.05))
        self.theta_min = float(cfg.get("theta_min", 0.5))
        self.theta_max = float(cfg.get("theta_max", 1.5))
        if not 0 <= self.rho_fast < self.rho_slow < 1:
            raise ValueError("Require 0 <= rho_fast < rho_slow < 1")
        if not self.theta_min < 1.0 < self.theta_max:
            raise ValueError("Threshold bounds must contain the initial threshold 1")
        if not 0 <= self.target_rate <= 1:
            raise ValueError("target_rate must be in [0, 1]")

        initial_probability = (1.0 - self.theta_min) / (self.theta_max - self.theta_min)
        self.threshold_logits = Parameter(self.weight.new_full(
            (self.out_features,), math.log(initial_probability / (1 - initial_probability))
        ))
        self.homeostatic_gain = Parameter(self.weight.new_zeros(()))
        self.transient_gain = Parameter(self.weight.new_zeros(()))

        def step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
            u_prev, z_prev, w_prev, fast_prev, slow_prev = carry
            activity = z_prev.mean(dim=-1, keepdim=True)
            fast = self.rho_fast * fast_prev + (1 - self.rho_fast) * activity
            slow = self.rho_slow * slow_prev + (1 - self.rho_slow) * activity
            threshold = self.dynamic_threshold(fast, slow)
            if self.use_recurrent:
                cur = cur + F.linear(z_prev, recurrent)
            u = alpha * u_prev + (1 - alpha) * (cur - w_prev)
            z = spike_grad_injection_function(u - threshold, self.alpha, self.c)
            u = u * (1 - z.detach()) + u_rest * z.detach()
            w = beta * w_prev + (1 - beta) * (a * u + b * z) * self.q
            return (u, z, w, fast, slow), z

        self.step = step_fn

    def dynamic_threshold(self, fast, slow):
        logits = (
            self.threshold_logits
            + self.homeostatic_gain * (slow - self.target_rate)
            - self.transient_gain * (fast - slow).abs()
        )
        return self.theta_min + (self.theta_max - self.theta_min) * logits.sigmoid()

    def initial_state(self, batch_size, device=None):
        u, z, w = super().initial_state(batch_size, device)
        # Equal initial traces avoid an artificial transient at sequence start.
        # Full-sized traces also support generic_scan_with_states stacking.
        rate = torch.full_like(z, self.target_rate, requires_grad=False)
        return u, z, w, rate, rate.clone()

    def apply_parameter_constraints(self):
        super().apply_parameter_constraints()
        with torch.no_grad():
            self.homeostatic_gain.clamp_(0, 5)
            self.transient_gain.clamp_(0, 5)

    def _scan(self, inputs, with_states=False):
        current = F.linear(inputs, self.weight, self.bias)
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        initial = self.initial_state(inputs.shape[0], inputs.device)
        u_rest = self.u0 if self.use_u_rest else torch.zeros_like(self.u0)

        def step(carry, cur):
            return self.step(self.recurrent, decay_u, decay_w, self.thr,
                             self.a, self.b, u_rest, carry, cur)

        scan = generic_scan_with_states if with_states else generic_scan
        return scan(step, initial, current, self.unroll)

    def layer_forward(self, inputs):
        return self._scan(inputs)[..., :self.num_out_neuron]

    @torch.no_grad()
    def layer_forward_with_states(self, inputs):
        states, outputs = self._scan(inputs, with_states=True)
        return states[..., :self.num_out_neuron], outputs[..., :self.num_out_neuron]
