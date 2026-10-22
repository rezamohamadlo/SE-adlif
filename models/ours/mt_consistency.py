"""An isolated training objective; MT inference and checkpoint keys are unchanged."""

import math

from functional.temporal_consistency import temporal_js_loss
from models.pl_module import MLPSNN


class MTConsistencyMLPSNN(MLPSNN):
    def __init__(self, cfg):
        super().__init__(cfg)
        if self.auto_regression or any(layer.cell != 'mt_se_adlif'
                                       for layer in [cfg.l1, cfg.l2]):
            raise ValueError('This experiment requires two-layer MT classification')
        self.consistency_weight = float(cfg.temporal_consistency.weight)
        window = cfg.temporal_consistency.window
        if (isinstance(window, bool) or not isinstance(window, int) or window < 1
                or not math.isfinite(self.consistency_weight) or self.consistency_weight < 0):
            raise ValueError('Require nonnegative finite weight and positive integer window')
        self.consistency_window = window

    def training_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)
        reduced, supervised, _ = self.process_predictions_and_compute_losses(
            outputs, targets, block_idx)
        self.update_and_log_metrics(reduced, targets, supervised, self.train_metric, 'train_')
        if self.consistency_weight == 0:
            return supervised
        consistency = temporal_js_loss(outputs, targets, block_idx,
                                       self.consistency_window, self.ignore_target_idx)
        total = supervised + self.consistency_weight * consistency
        self.log('train_temporal_js', consistency, on_step=True, on_epoch=True)
        self.log('train_total_loss', total, on_step=True, on_epoch=True)
        return total
