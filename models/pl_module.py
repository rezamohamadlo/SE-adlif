import math
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss, MSELoss
from omegaconf import DictConfig

from models.alif import EFAdLIF, SEAdLIF
from models.ours.dth_adlif import DTHSEAdLIF
from models.ours.mt_adlif import MTSEAdLIF
from models.ours.mr_adlif import MRSEAdLIF
from models.ours.fp_adlif import FPSEAdLIF
from models.ours.da_adlif import DASEAdLIF
from models.li import LI
from models.lif import LIF
from models.rnn import LSTMCellWrapper


layer_map = {
    "lif": LIF,
    "se_adlif": SEAdLIF,
    "dth_se_adlif": DTHSEAdLIF,
    "mt_se_adlif": MTSEAdLIF,
    "mr_se_adlif": MRSEAdLIF,
    "fp_se_adlif": FPSEAdLIF,
    "da_se_adlif": DASEAdLIF,
    "ef_adlif": EFAdLIF,
    'lstm': LSTMCellWrapper,
}


def aggregate_temporal_outputs(
    outputs: torch.Tensor,
    block_idx: torch.Tensor,
    num_blocks: int,
    loss_agg: str,
) -> torch.Tensor:
    """Aggregate temporal model outputs into target-aligned blocks."""
    if loss_agg in ("sum_softmax_over_time", "softmax"):
        # ``softmax`` is the legacy name retained for existing experiment
        # configurations. Both names mean softmax per timestep, then sum.
        values = torch.softmax(outputs, dim=-1)
        reduction = "sum"
    elif loss_agg == "summed_membrane_potentials":
        values = outputs
        reduction = "sum"
    elif loss_agg == "mean_membrane_potentials":
        values = outputs
        reduction = "mean"
    else:
        raise ValueError(
            f"Unsupported loss_agg {loss_agg!r}. Expected one of: "
            "'sum_softmax_over_time', 'summed_membrane_potentials', "
            "'mean_membrane_potentials', or legacy 'softmax'."
        )

    block_outputs = torch.zeros(
        size=(outputs.size(0), num_blocks, outputs.size(2)),
        dtype=outputs.dtype,
        device=outputs.device,
    )
    expanded_block_idx = block_idx.unsqueeze(-1).expand_as(outputs)
    return torch.scatter_reduce(
        block_outputs,
        dim=1,
        index=expanded_block_idx,
        src=values,
        reduce=reduction,
        include_self=False,
    )


class MLPSNN(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        print(cfg)
        self.ignore_target_idx = -1
        self.two_layers = cfg.two_layers
        self.output_size = cfg.dataset.num_classes
        self.tracking_metric = cfg.tracking_metric
        self.tracking_mode = cfg.tracking_mode
        self.batch_size = cfg.dataset.batch_size
        self.dropout = cfg.dropout


        # For learning rate scheduling (used for oscillation task)
        self.lr = cfg.lr
        self.factor = cfg.factor
        self.patience = cfg.patience

        self.auto_regression =  cfg.get('auto_regression', False)

        # Define the model
        self.l1 = layer_map[cfg.l1.cell](cfg.l1)
        if cfg.two_layers:
            self.l2 = layer_map[cfg.l2.cell](cfg.l2)
        self.out_layer = LI(cfg.l_out)
        
        self.output_func = cfg.get('loss_agg', 'softmax')
        self.init_metrics_and_loss()
        self.save_hyperparameters()

    def forward(
        self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.l1.initial_state(inputs.shape[0], inputs.device)
        s_out = self.out_layer.initial_state(inputs.shape[0], inputs.device)
        if self.two_layers:
            s2 = self.l2.initial_state(inputs.shape[0], inputs.device)
        out_sequence = []
        single_step_prediction_limit = int(math.ceil(inputs.shape[1] * 0.5))

        # Iterate over each time step in the data
        for t, x_t in enumerate(inputs.unbind(1)):

            # Auto-regression for oscillator task
            if self.auto_regression and t >= single_step_prediction_limit:
                x_t = out.detach()
            out, s1 = self.l1(x_t, s1)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            if self.two_layers:
                out, s2 = self.l2(out, s2)
                out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            out, s_out = self.out_layer(out, s_out)
            out_sequence.append(out)
            
        return torch.stack(out_sequence, dim=1)

    def on_train_batch_end(self, outputs, batch, batch_idx: int):
        self.l1.apply_parameter_constraints()
        if self.two_layers:
            self.l2.apply_parameter_constraints()
        self.out_layer.apply_parameter_constraints()

    def process_predictions_and_compute_losses(self, outputs, targets, block_idx):
        """
        Process the model output into prediction
        with respect to the temporal segmentation defined by the
        block_idx tensor.
        Then compute losses
        Args:
            outputs (torch.Tensor): full outputs
            targets (torch.Tensor): targets
            block_idx (torch.Tensor): tensor of index that determined which temporal segements of
            output time-step depends on which specific target,
            used by the scatter reduce operation.

        Returns:
            (): _description_
        """
        # compute softmax for every time-steps with respect to
        # the number of class
        if self.auto_regression:
            targets = targets[:, 1:]
            l2_loss = (outputs - targets) ** 2
            
            block_outputs = torch.zeros(
                size=(targets.shape[0], 2, outputs.shape[2]),
                dtype=outputs.dtype,
                device=outputs.device,
            )
            _block_idx = block_idx.unsqueeze(2).expand(size=(-1, -1, outputs.size(2)))
            block_output = torch.scatter_reduce(
                block_outputs,
                dim=1,
                index=_block_idx,
                src=l2_loss,
                reduce="mean",
                include_self=False,
            )
            block_output = block_output[:, 1]
            outputs_reduce = outputs
            loss = block_output.mean()
        else:
            block_output = aggregate_temporal_outputs(
                outputs=outputs,
                block_idx=block_idx,
                num_blocks=targets.size(1),
                loss_agg=self.output_func,
            )
            block_idx = block_idx.unsqueeze(-1)


            outputs_reduce = block_output.reshape(-1, outputs.size(-1))
            targets_reduce = targets.flatten()

            block_mask = torch.where(targets_reduce != self.ignore_target_idx)

            loss = self.loss(outputs_reduce[block_mask].float(), targets_reduce[block_mask])
        return (outputs_reduce, loss, block_idx)

    def update_and_log_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss: float,
        metrics: torchmetrics.MetricCollection,
        prefix: str,
    ):
        """
        Method centralizing the metrics logging mecanisms.

        Args:
            outputs_reduce (torch.Tensor): output prediction
            targets_reduce (torch.Tensor): target
            loss (float): loss
            metrics (torchmetrics.MetricCollection): collection of torchmetrics metrics
            aux_metrics (dict): auxiliary metrics that do not
            fit the torchmetrics logic
            prefix (str): prefix defining the stage of model either
            "train_": training stage
            "val_": validation stage
            "test_": testing stage
            Those prefix prevent clash of names in the logger.

        """
        if self.auto_regression:
            single_step_prediction_limit = int(math.ceil(0.5*outputs.shape[1]))
            outputs = outputs[:, single_step_prediction_limit:].squeeze()
            targets = targets[:, single_step_prediction_limit+1:].squeeze()
            outputs = outputs.reshape(-1, outputs.shape[-1])
            targets = targets.reshape(-1, targets.shape[-1])
        else:
            targets = targets.flatten()

        metrics(outputs, targets)
        self.log_dict(
            metrics,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )
        self.log(
            f"{prefix}loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True if prefix == "train_" else False,
        )

    def training_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(
            inputs,
        )
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.train_metric,
            prefix="train_",
        )

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)
        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.val_metric,
            prefix="val_",
        )

        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets, block_idx = batch
        outputs = self(inputs)

        (
            outputs_reduce,
            loss,
            block_idx,
        ) = self.process_predictions_and_compute_losses(outputs, targets, block_idx)

        self.update_and_log_metrics(
            outputs_reduce,
            targets,
            loss,
            self.test_metric,
            prefix="test_",
        )

        return loss

    def init_metrics_and_loss(self):
        if self.auto_regression:
            metrics = torchmetrics.MetricCollection(
                {
                    "mse": torchmetrics.MeanSquaredError(),
                }
            )
            self.loss = MSELoss()
        else:
            metrics = torchmetrics.MetricCollection(
                {
                    "acc": torchmetrics.Accuracy(
                        task="multiclass",  # type: ignore
                        num_classes=self.output_size,
                        average="micro",
                        ignore_index=self.ignore_target_idx,
                    )
                }
            )
            self.loss = CrossEntropyLoss(ignore_index=self.ignore_target_idx)
        self.train_metric = metrics.clone(prefix="train_")
        self.val_metric = metrics.clone(prefix="val_")
        self.test_metric = metrics.clone(prefix="test_")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=self.tracking_mode,
            factor=self.factor,
            patience=self.patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": self.tracking_metric,
            },
        }
