"""Explicit train-only objective; ordinary neuron/readout behavior is retained."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.pl_module import MLPSNN
from models.ours.input_fir import InputFIRMLPSNN

OBJECTIVE = 'classification_label_smoothing_v1'


def validate_label_smoothing_config(cfg):
    value = cfg.get('label_smoothing_epsilon')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
        raise ValueError('Active label smoothing requires finite 0 < epsilon < 1')
    if cfg.get('auto_regression', False):
        raise ValueError('Label smoothing is classification-only')
    if cfg.get('spike_delete_probability', 0):
        raise ValueError('Label smoothing plus event deletion requires a separate declared study')
    if cfg.get('input_fir') and cfg.get('input_fir').get('mode', 'causal') != 'gain_only':
        raise ValueError('This objective supports the declared gain-only combination, not causal FIR')
    return float(value)


class TrainingOnlyLabelSmoothingLoss(nn.CrossEntropyLoss):
    """Keep integer targets/masks; eval uses exactly ordinary hard-target CE."""
    def __init__(self, epsilon, ignore_index=-1):
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or not 0 <= epsilon < 1:
            raise ValueError('Require finite 0 <= epsilon < 1')
        super().__init__(ignore_index=ignore_index)
        self.epsilon = float(epsilon)
        self.last_hard_ce = None  # Diagnostic scalar, neither parameter nor buffer.

    def forward(self, inputs, targets):
        if not self.training or self.epsilon == 0:
            self.last_hard_ce = None
            return super().forward(inputs, targets)
        with torch.no_grad():
            self.last_hard_ce = F.cross_entropy(inputs.detach(), targets,
                                               ignore_index=self.ignore_index)
        return F.cross_entropy(inputs, targets, ignore_index=self.ignore_index,
                               label_smoothing=self.epsilon)


class _LabelSmoothingObjective:
    def __init__(self, cfg):
        epsilon = validate_label_smoothing_config(cfg)
        super().__init__(cfg)
        self.loss = TrainingOnlyLabelSmoothingLoss(epsilon, self.ignore_target_idx)

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        hard_ce = self.loss.last_hard_ce
        if hard_ce is not None:
            self.log('train_hard_ce', hard_ce, on_step=False, on_epoch=True,
                     batch_size=batch[0].shape[0], prog_bar=False)
        return loss


class LabelSmoothedMLPSNN(_LabelSmoothingObjective, MLPSNN):
    pass


class LabelSmoothedInputGainMLPSNN(_LabelSmoothingObjective, InputFIRMLPSNN):
    pass


def build_label_smoothed_model(cfg):
    cls = LabelSmoothedInputGainMLPSNN if cfg.get('input_fir') else LabelSmoothedMLPSNN
    return cls(cfg)
