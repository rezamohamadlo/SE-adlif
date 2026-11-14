"""Opt-in R-Drop training; ordinary single-pass inference is retained."""
import math

import torch
from models.pl_module import MLPSNN
from models.ours.label_smoothing import TrainingOnlyLabelSmoothingLoss

OBJECTIVE = 'classification_dropout_consistency_v1'


def validate_dropout_consistency_config(cfg):
    alpha = cfg.get('dropout_consistency_alpha')
    epsilon = cfg.get('label_smoothing_epsilon', 0.)
    for name, value in (('alpha', alpha), ('epsilon', epsilon)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Dropout consistency {name} must be finite numeric')
    if alpha < 0 or not 0 <= epsilon < 1:
        raise ValueError('Require alpha >= 0 and 0 <= smoothing epsilon < 1')
    backbone = cfg.get('dropout_consistency_backbone', 'se_adlif')
    if (backbone not in ('se_adlif', 'dth_se_adlif')
            or cfg.get('auto_regression', False) or not cfg.two_layers
            or any(cfg[layer].cell != backbone for layer in ('l1', 'l2'))):
        raise ValueError('Require a declared matching two-layer SE or explicit DTH classification backbone')
    if not 0 < cfg.dropout < 1 or cfg.get('input_fir') or cfg.get('spike_delete_probability', 0):
        raise ValueError('Require ordinary dropout and no undeclared input transforms')
    return float(alpha), float(epsilon)


def symmetric_kl_from_logits(first, second):
    """Class-sum, sample-mean symmetric KL with gradients through both views."""
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] == 0:
        raise ValueError('Require matching nonempty [valid blocks, classes] logits')
    log_first = first.log_softmax(-1)
    log_second = second.log_softmax(-1)
    return .5 * ((log_first.exp()-log_second.exp()) * (log_first-log_second)).sum(-1).mean()


class DropoutConsistentMLPSNN(MLPSNN):
    def __init__(self, cfg):
        alpha, epsilon = validate_dropout_consistency_config(cfg)
        super().__init__(cfg)
        self.consistency_alpha = alpha
        self.loss = TrainingOnlyLabelSmoothingLoss(epsilon, self.ignore_target_idx)

    def training_terms(self, batch):
        """Two independent dropout graphs, with the unchanged target/block masks."""
        if not self.training:
            raise ValueError('Two-pass objective is training-only')
        inputs, targets, blocks = batch
        first, loss_first, _ = self.process_predictions_and_compute_losses(self(inputs), targets, blocks)
        hard_first = self.loss.last_hard_ce if self.loss.last_hard_ce is not None else loss_first.detach()
        second, loss_second, _ = self.process_predictions_and_compute_losses(self(inputs), targets, blocks)
        hard_second = self.loss.last_hard_ce if self.loss.last_hard_ce is not None else loss_second.detach()
        valid = targets.flatten().ne(self.ignore_target_idx)
        consistency = symmetric_kl_from_logits(first[valid].float(), second[valid].float())
        supervised = .5*(loss_first+loss_second)
        total = supervised+self.consistency_alpha*consistency
        return first, total, {'train_supervised_loss': supervised.detach(),
                            'train_dropout_symmetric_kl': consistency.detach(),
                            'train_hard_ce': .5*(hard_first+hard_second)}

    def training_step(self, batch, batch_idx):
        first, total, diagnostics = self.training_terms(batch)
        self.update_and_log_metrics(first, batch[1], total, self.train_metric, 'train_')
        for name, value in diagnostics.items():
            self.log(name, value, on_step=False, on_epoch=True,
                     batch_size=batch[0].shape[0], prog_bar=False)
        return total
