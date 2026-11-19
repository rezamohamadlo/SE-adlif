"""Opt-in residual per-frame current normalization with an explicit gain control."""
import math

import torch
from torch.nn import Parameter
from torch.nn import functional as F

from models.alif import SEAdLIF


class ResidualCurrentNormSEAdLIF(SEAdLIF):
    """Condition feed-forward current; keep SE state, recurrence and reset intact.

    Statistics use only the last feature axis of one sample/frame. Removing
    bias before the non-affine branch preserves the original current on zero
    input. The zero-initialized scalar gate keeps the initial reference limit.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        modes = {'residual_current_norm_se_adlif': 'layer_norm',
                 'current_gain_se_adlif': 'gain_only'}
        expected = modes.get(cfg.get('cell'))
        mode = cfg.get('current_norm_mode', expected or 'layer_norm')
        epsilon = cfg.get('current_norm_eps', 1e-5)
        if mode not in ('layer_norm', 'gain_only') or (expected and mode != expected):
            raise ValueError('Current conditioning mode must match its registered cell')
        if (isinstance(epsilon, bool) or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError('Current normalization epsilon must be finite and positive')
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.current_norm_mode = mode
        self.current_norm_eps = float(epsilon)
        self.current_norm_gate = Parameter(self.weight.new_zeros(()))

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, 'current_norm_gate'):
            with torch.no_grad():
                self.current_norm_gate.zero_()

    def feedforward_current(self, inputs):
        current = F.linear(inputs, self.weight, self.bias)
        bias_free = current-self.bias if self.bias is not None else current
        branch = (F.layer_norm(bias_free, (self.out_features,),
                              weight=None, bias=None, eps=self.current_norm_eps)
                  if self.current_norm_mode == 'layer_norm' else bias_free)
        return current+self.current_norm_gate.tanh()*branch

    def forward(self, input_tensor, states):
        # Match the inherited single-step API, including its resting-potential
        # convention. No change to the parent SE integration step is made.
        current = self.feedforward_current(input_tensor)
        new_states, spikes = self.step(self.recurrent,
            self.tau_u_trainer.get_decay(), self.tau_w_trainer.get_decay(),
            self.thr, self.a, self.b, self.u0, states, current)
        return spikes, new_states

    def layer_forward(self, inputs):
        current = self.feedforward_current(inputs)
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        out = self.wrapped_scan(u, z, w, current, self.recurrent,
            self.tau_u_trainer.get_decay(), self.tau_w_trainer.get_decay(),
            self.thr, self.a, self.b)
        return out[:, :, :self.num_out_neuron]

    @torch.no_grad()
    def layer_forward_with_states(self, inputs):
        current = self.feedforward_current(inputs)
        u, z, w = self.initial_state(int(inputs.shape[0]), inputs.device)
        states, out = self.wrapped_scan_with_states(u, z, w, current, self.recurrent,
            self.tau_u_trainer.get_decay(), self.tau_w_trainer.get_decay(),
            self.thr, self.a, self.b)
        return states[..., :self.num_out_neuron], out[..., :self.num_out_neuron]
