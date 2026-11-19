"""DTH with learned, bounded layer-level activity-trace decay rates."""

import math

import torch.nn.functional as F
from torch.nn import Parameter

from models.helpers import spike_grad_injection_function
from models.ours.dth_adlif import DTHSEAdLIF


class LTDTHSEAdLIF(DTHSEAdLIF):
    """Learn two trace rates per layer, keeping DTH's neuron dynamics intact.

    Trace decays are per input frame, not physical millisecond constants.
    Nonoverlapping ranges guarantee 0 < rho_fast < rho_slow < 1 even when
    sigmoid saturates. Initial rates match the inherited DTH configuration.
    Activity remains causal and sample-local; learned rates are shared within
    a layer, not across samples' activity states. State order matches DTH.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.trace_rho_fast_range = self._bounds(
            cfg.get('trace_rho_fast_range', [0.5, 0.98]), 'fast')
        self.trace_rho_slow_range = self._bounds(
            cfg.get('trace_rho_slow_range', [0.985, 0.9995]), 'slow')
        if not self.trace_rho_fast_range[1] < self.trace_rho_slow_range[0]:
            raise ValueError('Fast and slow trace ranges must be strictly separated')
        self.trace_fast_logit = Parameter(self.weight.new_tensor(
            self._initial_logit(self.rho_fast, self.trace_rho_fast_range)))
        self.trace_slow_logit = Parameter(self.weight.new_tensor(
            self._initial_logit(self.rho_slow, self.trace_rho_slow_range)))

        def step_fn(recurrent, alpha, beta, thr, a, b, u_rest, carry, cur):
            u_prev, z_prev, w_prev, fast_prev, slow_prev = carry
            rho_fast, rho_slow = self.trace_decays()
            activity = z_prev.mean(dim=-1, keepdim=True)
            fast = rho_fast * fast_prev + (1 - rho_fast) * activity
            slow = rho_slow * slow_prev + (1 - rho_slow) * activity
            threshold = self.dynamic_threshold(fast, slow)
            if self.use_recurrent:
                cur = cur + F.linear(z_prev, recurrent)
            u = alpha * u_prev + (1 - alpha) * (cur - w_prev)
            z = spike_grad_injection_function(u - threshold, self.alpha, self.c)
            u = u * (1 - z.detach()) + u_rest * z.detach()
            w = beta * w_prev + (1 - beta) * (a * u + b * z) * self.q
            return (u, z, w, fast, slow), z

        self.step = step_fn

    @staticmethod
    def _bounds(values, name):
        if (len(values) != 2 or not all(math.isfinite(float(x)) for x in values)
                or not 0 < values[0] < values[1] < 1):
            raise ValueError(f'{name} trace range must satisfy 0 < min < max < 1')
        return tuple(float(x) for x in values)

    @staticmethod
    def _initial_logit(rate, bounds):
        low, high = bounds
        if not low < rate < high:
            raise ValueError('Initial DTH trace rate must be strictly inside its learned range')
        return math.log((rate - low) / (high - rate))

    def trace_decays(self):
        fast_low, fast_high = self.trace_rho_fast_range
        slow_low, slow_high = self.trace_rho_slow_range
        return (fast_low + (fast_high - fast_low) * self.trace_fast_logit.sigmoid(),
                slow_low + (slow_high - slow_low) * self.trace_slow_logit.sigmoid())
