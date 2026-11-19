"""Opt-in channel-specific causal filtering; unchanged SE-adLIF neuron dynamics."""
import torch
from torch import nn
from models.pl_module import MLPSNN
from functional.count_thinning import thin_input_counts


def validate_fir(config):
    lags = list(config.get('lags', [1, 2, 4, 8]))
    if not lags or len(set(lags)) != len(lags) or any(type(x) is not int or x < 1 for x in lags):
        raise ValueError('FIR lags must be unique positive frame counts')
    return lags


class CausalInputFIR(nn.Module):
    def __init__(self, channels, lags, mode='causal'):
        super().__init__()
        if mode not in ('causal', 'gain_only'):
            raise ValueError('Input FIR mode must be causal or gain_only')
        self.mode = mode
        self.lags = validate_fir(dict(lags=lags))
        self.coefficients = nn.Parameter(torch.zeros(len(self.lags), channels))

    def forward(self, inputs):
        if self.mode == 'gain_only':
            # Parameter-matched DC gain control, with no temporal history.
            return inputs * (1 + self.coefficients.tanh().sum(dim=0))
        result = inputs
        for lag, coefficient in zip(self.lags, self.coefficients.tanh()):
            if lag < inputs.shape[1]:
                past = torch.nn.functional.pad(inputs[:, :-lag, :], (0, 0, lag, 0))
                result = result + coefficient * past
        return result


class InputFIRMLPSNN(MLPSNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.input_fir = CausalInputFIR(cfg.l1.input_size, validate_fir(cfg.input_fir),
                                     cfg.input_fir.get('mode', 'causal'))

    def forward(self, inputs):
        inputs=thin_input_counts(inputs,self.hparams.cfg.get('spike_delete_probability',0),self.training)
        return super().forward(self.input_fir(inputs))
