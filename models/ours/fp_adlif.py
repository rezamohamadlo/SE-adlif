"""Frequency-parameterized SE-adLIF; dt and tau are in milliseconds."""

import math

import torch
import torch.nn.functional as F

from models.alif import SEAdLIF


class FPSEAdLIF(SEAdLIF):
    """Learn bounded per-neuron frequencies instead of direct membrane coupling.

    State remains (u, spikes, w). The inherited SE step/reset/surrogate is
    unchanged. Its q*a coefficient is the physical frequency-derived coupling.
    Frequency describes the autonomous, spike-free single-neuron subsystem,
    not the spiking or recurrent network's dominant output frequency.
    """

    def __init__(self, cfg, device=None, dtype=None, **kwargs):
        dt = float(cfg.get("dt", 1.0))
        bounds = cfg.get("frequency_range_hz", [1.0, 50.0])
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive (milliseconds)")
        if (len(bounds) != 2 or not all(math.isfinite(float(x)) for x in bounds)
                or not 0 < bounds[0] < bounds[1] < 500.0 / dt):
            raise ValueError("frequency_range_hz must satisfy 0 < min < max < Nyquist (500/dt_ms)")
        if not math.isfinite(float(cfg.q)) or cfg.q <= 0:
            raise ValueError("q must be finite and positive")
        for name in ("tau_u_range", "tau_w_range"):
            values = cfg[name]
            if (len(values) != 2 or not all(math.isfinite(float(x)) for x in values)
                    or not 0 < values[0] <= values[1]):
                raise ValueError(f"{name} must be finite, positive and ordered")
        super().__init__(cfg, device=device, dtype=dtype, **kwargs)
        self.dt_seconds = dt * 1e-3
        self.register_buffer("frequency_min_hz", self.weight.new_tensor(float(bounds[0])))
        self.register_buffer("frequency_max_hz", self.weight.new_tensor(float(bounds[1])))
        # Reuse the parameter allocation, but remove direct a from the optimizer.
        raw = self.a
        del self.a
        self.frequency_logits = raw
        torch.nn.init.uniform_(self.frequency_logits, -2.0, 2.0)

    @property
    def frequency_hz(self):
        return self.frequency_min_hz + (
            self.frequency_max_hz - self.frequency_min_hz
        ) * self.frequency_logits.sigmoid()

    def derived_a(self, alpha=None, beta=None):
        """Return repository-scaled a, retaining gradients through f and tau.

        Rewrite the numerator to avoid subtracting nearly equal cosines:
        (sqrt(alpha)-sqrt(beta))**2 + 4*sqrt(alpha*beta)*sin(phi/2)**2.
        Promote half precision for this calculation; protect its denominator.
        """
        alpha = self.tau_u_trainer.get_decay() if alpha is None else alpha
        beta = self.tau_w_trainer.get_decay() if beta is None else beta
        dtype = torch.float64 if alpha.dtype == torch.float64 else torch.float32
        a_decay, b_decay = alpha.to(dtype), beta.to(dtype)
        root_a, root_b = a_decay.sqrt(), b_decay.sqrt()
        half_phi = math.pi * self.frequency_hz.to(dtype) * self.dt_seconds
        numerator = (root_a - root_b).square() + 4 * root_a * root_b * half_phi.sin().square()
        denominator = ((1-a_decay) * (1-b_decay)).clamp_min(torch.finfo(dtype).eps)
        return (numerator / denominator / self.q).to(alpha.dtype)

    def initial_state(self, batch_size, device=None):
        zeros = self.weight.new_zeros((batch_size, self.out_features), device=device)
        return self.u0.unsqueeze(0), zeros, zeros.clone()

    def forward(self, input_tensor, states):
        alpha, beta = self.tau_u_trainer.get_decay(), self.tau_w_trainer.get_decay()
        current = F.linear(input_tensor, self.weight, self.bias)
        state, spikes = self.step(self.recurrent, alpha, beta, self.thr,
                                 self.derived_a(alpha, beta), self.b, self.u0, states, current)
        return spikes, state

    def _scan(self, inputs, with_states=False):
        alpha, beta = self.tau_u_trainer.get_decay(), self.tau_w_trainer.get_decay()
        scan = self.wrapped_scan_with_states if with_states else self.wrapped_scan
        return scan(*self.initial_state(inputs.shape[0], inputs.device),
                    F.linear(inputs, self.weight, self.bias), self.recurrent,
                    alpha, beta, self.thr, self.derived_a(alpha, beta), self.b)

    def layer_forward(self, inputs):
        return self._scan(inputs)[..., :self.num_out_neuron]

    @torch.no_grad()
    def layer_forward_with_states(self, inputs):
        states, spikes = self._scan(inputs, with_states=True)
        return states[..., :self.num_out_neuron], spikes[..., :self.num_out_neuron]

    def apply_parameter_constraints(self):
        # No inherited a clamp: a is derived, not an independently learned value.
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        with torch.no_grad():
            self.b.clamp_(*self.b_range)
            self.u0.copy_(self.u0 - self.u0.sign() * torch.relu(self.u0.abs() - self.thr))
            self.thr.clamp_(min=0)

    def reset_parameters(self):
        # During parent construction the original a parameter still exists.
        if "frequency_logits" not in self._parameters:
            return super().reset_parameters()
        self.tau_u_trainer.reset_parameters()
        self.tau_w_trainer.reset_parameters()
        torch.nn.init.uniform_(self.weight, -self.ff_gain / math.sqrt(self.in_features),
                               self.ff_gain / math.sqrt(self.in_features))
        torch.nn.init.zeros_(self.bias)
        if self.train_u0:
            torch.nn.init.uniform_(self.u0, 0, self.thr[0].item())
        else:
            torch.nn.init.zeros_(self.u0)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(self.recurrent)
        torch.nn.init.uniform_(self.b, *self.b_range)
        torch.nn.init.uniform_(self.frequency_logits, -2.0, 2.0)
