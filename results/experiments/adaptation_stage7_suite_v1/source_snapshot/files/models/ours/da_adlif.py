"""Delay-aware SE-adLIF with a gated causal mixture of fixed delays."""

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from models.alif import SEAdLIF
from models.helpers import generic_scan


class DASEAdLIF(SEAdLIF):
    """Filter projected feedforward current, preserving the original SE step.

    Each neuron learns a gate and softmax weights over fixed past-frame delays.
    The zero-initialized gate reproduces SE-adLIF exactly; mixture gradients
    begin once the gate moves away from zero. Recurrent current is not delayed.
    State is (u, spikes, w, history), with history [batch, max_delay, neurons],
    newest first. History is zero-padded before sequence start and never detached.
    Delays are input frames (SHD frames are 4 ms), not the cell's configured dt.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        delays = cfg.get("delay_frames", [1, 2, 4])
        if (not delays or any(isinstance(d, bool) or not isinstance(d, int) or d < 1
                              for d in delays) or len(set(delays)) != len(delays)):
            raise ValueError("delay_frames must contain distinct positive integers")
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.delay_frames = tuple(delays)
        self.max_delay = max(delays)
        self.register_buffer("delay_indices", torch.tensor(
            [d-1 for d in delays], device=self.weight.device, dtype=torch.long))
        self.delay_gate = Parameter(self.weight.new_zeros(self.out_features))
        self.delay_logits = Parameter(self.weight.new_zeros(self.out_features, len(delays)))

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self, "delay_gate"):
            torch.nn.init.zeros_(self.delay_gate)
            torch.nn.init.zeros_(self.delay_logits)

    def initial_state(self, batch_size, device=None):
        zeros = self.weight.new_zeros((batch_size, self.out_features), device=device)
        history = self.weight.new_zeros(
            (batch_size, self.max_delay, self.out_features), device=device)
        return self.u0.unsqueeze(0), zeros, zeros.clone(), history

    def filter_current(self, current, history):
        weights = self.delay_logits.softmax(dim=-1).transpose(0, 1)
        delayed = (history.index_select(1, self.delay_indices) * weights.unsqueeze(0)).sum(1)
        gate = self.delay_gate.clamp(0, 1)
        filtered = (1-gate)*current + gate*delayed
        # Read only past frames above, then push the current projected input.
        history = torch.cat((current.unsqueeze(1), history[:, :-1]), dim=1)
        return filtered, history

    def _delayed_step(self, alpha, beta, rest, state, current):
        filtered, history = self.filter_current(current, state[3])
        neuron_state, spikes = self.step(self.recurrent, alpha, beta, self.thr,
                                        self.a, self.b, rest, state[:3], filtered)
        return (*neuron_state, history), spikes

    def forward(self, input_tensor, states):
        current = F.linear(input_tensor, self.weight, self.bias)
        state, spikes = self._delayed_step(self.tau_u_trainer.get_decay(),
                                          self.tau_w_trainer.get_decay(),
                                          self.u0, states, current)
        return spikes, state

    def _sequence_step(self):
        alpha = self.tau_u_trainer.get_decay()
        beta = self.tau_w_trainer.get_decay()
        rest = self.u0 if self.use_u_rest else torch.zeros_like(self.u0)

        def step(state, current):
            return self._delayed_step(alpha, beta, rest, state, current)
        return step

    def layer_forward(self, inputs):
        current = F.linear(inputs, self.weight, self.bias)
        spikes = generic_scan(self._sequence_step(),
                              self.initial_state(inputs.shape[0], inputs.device),
                              current, self.unroll)
        return spikes[..., :self.num_out_neuron]

    @torch.no_grad()
    def layer_forward_with_states(self, inputs):
        """Return tuple histories: u/z/w [B,T+1,N], delay memory [B,T+1,D,N]."""
        state = self.initial_state(inputs.shape[0], inputs.device)
        step = self._sequence_step()
        histories, outputs = [[], [], [], []], []

        def snapshot():
            for target, value in zip(histories, state):
                target.append(value.expand(inputs.shape[0], *value.shape[1:]))

        snapshot()
        for current in F.linear(inputs, self.weight, self.bias).unbind(1):
            state, spikes = step(state, current)
            snapshot()
            outputs.append(spikes)
        return (tuple(torch.stack(values, dim=1)[..., :self.num_out_neuron]
                      for values in histories),
                torch.stack(outputs, dim=1)[..., :self.num_out_neuron])

    def apply_parameter_constraints(self):
        super().apply_parameter_constraints()
        with torch.no_grad():
            self.delay_gate.clamp_(0, 1)
