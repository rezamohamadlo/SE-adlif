import os
import tempfile
import csv
from pathlib import Path

# Some restricted Windows profiles cannot write Matplotlib's user cache.  Keep
# this process-local cache outside the project and in the system temp location.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "se-adlif-matplotlib")
)

import pytorch_lightning as pl
import hydra
import torch
from lightning_fabric.plugins.io import TorchCheckpointIO
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.csv_logs import CSVLogger, ExperimentWriter
import logging
from models.pl_module import MLPSNN

from pytorch_lightning.strategies import SingleDeviceStrategy


class TrustedLocalCheckpointIO(TorchCheckpointIO):
    """Load checkpoints written by this runner under PyTorch 2.6+.

    Lightning checkpoints include Hydra metadata in addition to tensor weights.
    This adapter is deliberately scoped to the runner's own checkpoint files,
    instead of changing PyTorch's global loading policy.
    """

    def load_checkpoint(self, path, map_location=lambda storage, loc: storage):
        return torch.load(path, map_location=map_location, weights_only=False)


class GradientNormProgressBar(pl.Callback):
    """Log the global L2 gradient norm before each optimizer step."""

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        parameter_norms = [
            parameter.grad.detach().norm(2)
            for parameter in pl_module.parameters()
            if parameter.grad is not None
        ]
        if not parameter_norms:
            return

        gradient_norm = torch.linalg.vector_norm(torch.stack(parameter_norms), ord=2)
        pl_module.log(
            "grad_norm",
            gradient_norm,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )


class ApplyLRSchedulerConfig(pl.Callback):
    """Make command-line scheduler settings authoritative after a resume."""

    def __init__(self, factor, patience):
        self.factor = factor
        self.patience = patience

    def on_train_start(self, trainer, pl_module):
        for scheduler_config in trainer.lr_scheduler_configs:
            scheduler = scheduler_config.scheduler
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.factor = self.factor
                scheduler.patience = self.patience


class TrainingSummaryCallback(pl.Callback):
    """Update TRAINING_SUMMARY.md whenever a new best checkpoint is saved."""

    def __init__(self, checkpoint_callback, cfg):
        self.checkpoint_callback = checkpoint_callback
        self.cfg = cfg
        self._summarized_best_path = None
        self._latest_validation_snapshot = None
        self._validation_sample_timesteps = 0
        self._validation_samples = 0

    @staticmethod
    def _metric(metrics, name):
        value = metrics.get(name)
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)

    @staticmethod
    def _format(value, digits=4, percent=False):
        if value is None:
            return "N/A"
        if percent:
            return f"{100 * value:.2f}%"
        return f"{value:.{digits}f}"

    def on_fit_start(self, trainer, pl_module):
        # Checkpoint state has been restored by this point. Do not rewrite the
        # summary unless a different best checkpoint is produced afterward.
        self._summarized_best_path = self.checkpoint_callback.best_model_path or None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._validation_sample_timesteps = 0
        self._validation_samples = 0

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if trainer.sanity_checking:
            return
        inputs = batch[0]
        batch_size, timesteps = inputs.shape[:2]
        self._validation_sample_timesteps += int(batch_size * timesteps)
        self._validation_samples += int(batch_size)

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        average_timesteps = (
            self._validation_sample_timesteps / self._validation_samples
            if self._validation_samples
            else None
        )

        # The SNN layers call torch.nn.functional.linear directly, so generic
        # module profilers do not see nn.Linear modules. Count their dense
        # weight and recurrent matrix multiply-adds explicitly. One MAC is
        # reported as two FLOPs. Elementwise neuron-state operations are not
        # included, which is stated in the generated summary.
        dense_macs_per_timestep = 0
        layers = [pl_module.l1, pl_module.out_layer]
        if pl_module.two_layers:
            layers.insert(1, pl_module.l2)
        for layer in layers:
            weight = getattr(layer, "weight", None)
            if weight is not None:
                dense_macs_per_timestep += weight.numel()
            recurrent = getattr(layer, "recurrent", None)
            if recurrent is not None and recurrent.ndim == 2:
                dense_macs_per_timestep += recurrent.numel()

        dense_forward_gflops = (
            2 * dense_macs_per_timestep * average_timesteps / 1e9
            if average_timesteps is not None
            else None
        )
        self._latest_validation_snapshot = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "val_acc": self._metric(metrics, "val_acc"),
            "val_loss": self._metric(metrics, "val_loss"),
            "train_acc": self._metric(metrics, "train_acc_epoch"),
            "train_loss": self._metric(metrics, "train_loss_epoch"),
            "grad_norm": self._metric(metrics, "grad_norm"),
            "average_timesteps": average_timesteps,
            "dense_forward_gflops": dense_forward_gflops,
            "total_parameters": sum(p.numel() for p in pl_module.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in pl_module.parameters() if p.requires_grad
            ),
        }

    def on_train_epoch_start(self, trainer, pl_module):
        # ModelCheckpoint saves at the end of validation, after ordinary
        # callbacks. At the next epoch start, the new file is guaranteed to
        # exist and can safely be referenced by the summary.
        self._write_if_new_best(trainer, status="Training in progress")

    def on_train_end(self, trainer, pl_module):
        self._write_if_new_best(trainer, status="Training finished")
        self._update_run_status(trainer, "Training finished")

    def on_test_end(self, trainer, pl_module):
        self._update_run_status(trainer, "Training and testing finished")
        self._update_test_results(trainer)

    def on_exception(self, trainer, pl_module, exception):
        self._write_if_new_best(trainer, status="Training interrupted; resumable")
        self._update_run_status(trainer, "Training interrupted; resumable")

    @staticmethod
    def _replace_summary(summary_path, text):
        temporary_path = summary_path.with_suffix(".md.tmp")
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, summary_path)

    def _update_run_status(self, trainer, status):
        summary_path = Path.cwd() / "TRAINING_SUMMARY.md"
        if not summary_path.exists():
            return

        completed_epochs = min(trainer.current_epoch + 1, int(self.cfg.n_epochs))
        final_epoch = max(completed_epochs - 1, 0)
        lines = summary_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.startswith("- **Status:**"):
                lines[index] = f"- **Status:** {status}"
            elif line.startswith("- **Progress:**"):
                lines[index] = (
                    f"- **Progress:** {completed_epochs} epochs completed "
                    f"(`0` through `{final_epoch}`) out of {self.cfg.n_epochs}"
                )
        self._replace_summary(summary_path, "\n".join(lines) + "\n")

    def _update_test_results(self, trainer):
        summary_path = Path.cwd() / "TRAINING_SUMMARY.md"
        if not summary_path.exists():
            return

        metrics = trainer.callback_metrics
        test_acc = self._metric(metrics, "test_acc")
        test_loss = self._metric(metrics, "test_loss")
        if test_acc is None and test_loss is None:
            return

        marker = "\n## Final test evaluation\n"
        summary = summary_path.read_text(encoding="utf-8")
        summary = summary.split(marker, 1)[0].rstrip()
        test_section = f"""

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **{self._format(test_acc, percent=True)}** |
| Test loss | **{self._format(test_loss)}** |
"""
        self._replace_summary(summary_path, summary + test_section)

    def _write_if_new_best(self, trainer, status):
        best_path = self.checkpoint_callback.best_model_path
        if not best_path or best_path == self._summarized_best_path:
            return
        if self._latest_validation_snapshot is None:
            return

        snapshot = self._latest_validation_snapshot
        summary_path = Path.cwd() / "TRAINING_SUMMARY.md"
        checkpoint_path = Path(best_path)
        try:
            checkpoint_display = checkpoint_path.resolve().relative_to(
                summary_path.parent.resolve()
            ).as_posix()
        except ValueError:
            checkpoint_display = checkpoint_path.as_posix()

        best_score = self.checkpoint_callback.best_model_score
        if isinstance(best_score, torch.Tensor):
            best_score = best_score.detach().cpu().item()

        learning_rate = trainer.optimizers[0].param_groups[0]["lr"]
        dataset_cfg = self.cfg.dataset
        dataset_name = self.cfg.exp_name.split("_", 1)[0].upper()
        validation_split = dataset_cfg.get("validate_on", "validation")
        early_stopping = self.cfg.get("early_stopping", False)
        mode = self.cfg.get("run_mode", "ours" if "DA_LIF" in self.cfg.exp_name else "reference")
        configured_logdir = Path(str(self.cfg.logdir)).as_posix().rstrip("/")
        seed_suffix = f"/{mode}/{dataset_name}/seed_{self.cfg.random_seed}"
        launcher_logdir = (
            configured_logdir[: -len(seed_suffix)]
            if configured_logdir.endswith(seed_suffix)
            else "results/phase1_models"
        )
        launcher_logdir = self.cfg.get("launcher_logdir", launcher_logdir)
        launcher_script = self.cfg.get("launcher_script", "run_phase1_se_adlif.ps1")
        dataset_seed_arguments = f" -Datasets {dataset_name} -Seeds {self.cfg.random_seed}"
        variant_argument = (
            f" -ModelVariant {self.cfg.model_variant}"
            if self.cfg.get("model_variant") else ""
        )
        if self.cfg.get("model_variant") == "MR_SE_adLIF":
            variant_argument += f" -AdaptationUpdateInterval {self.cfg.adaptation_update_interval}"
        resume_command = (
            f".\\.venv\\Scripts\\python.exe full_runner.py {self.cfg.model_variant}"
            + (f" --adaptation-update-interval {self.cfg.adaptation_update_interval}"
               if self.cfg.get("model_variant") == "MR_SE_adLIF" else "")
            if launcher_script == "full_runner.py" else
            f".\\{launcher_script} -Mode {mode}{variant_argument}{dataset_seed_arguments} "
            f"-Epochs {self.cfg.n_epochs} -BatchSize {dataset_cfg.batch_size} "
            f"-EarlyStopping {1 if early_stopping else 0} "
            f"-LrSchedulerPatience {self.cfg.patience} -LrSchedulerFactor {self.cfg.factor} "
            f"-Resume 1 -LogDir {launcher_logdir}"
        )

        summary = f"""# {self.cfg.exp_name} Training Summary — Seed {self.cfg.random_seed}

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** {status}
- **Progress:** {snapshot['epoch'] + 1} epochs completed (`0` through `{snapshot['epoch']}`) out of {self.cfg.n_epochs}
- **Dataset:** {dataset_name}
- **Model:** {self.cfg.exp_name}
- **Seed:** {self.cfg.random_seed}
- **Device:** {self.cfg.device}

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **{self._format(best_score, percent=True)}** | {snapshot['epoch']} |
| Validation loss | **{self._format(snapshot['val_loss'])}** | {snapshot['epoch']} |
| Training accuracy | **{self._format(snapshot['train_acc'], percent=True)}** | {snapshot['epoch']} |
| Training loss | **{self._format(snapshot['train_loss'], digits=6)}** | {snapshot['epoch']} |
| Gradient norm | **{self._format(snapshot['grad_norm'])}** | {snapshot['epoch']} |
| Learning rate | **{learning_rate:.8g}** | {snapshot['epoch']} |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **{snapshot['total_parameters']:,}** |
| Trainable parameters | **{snapshot['trainable_parameters']:,}** |
| Average padded validation timesteps/sample | **{self._format(snapshot['average_timesteps'], digits=2)}** |
| Estimated dense forward GFLOPs/sample | **{self._format(snapshot['dense_forward_gflops'], digits=4)}** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
{checkpoint_display}
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | {self.cfg.n_epochs} |
| Batch size | {dataset_cfg.batch_size} |
| Initial learning rate | {self.cfg.lr} |
| LR scheduler factor | {self.cfg.factor} |
| LR scheduler patience | {self.cfg.patience} |
| Early stopping | {str(bool(early_stopping)).lower()} |
| Early-stopping patience | {self.cfg.get('early_stopping_patience', 'N/A')} |
| Early-stopping minimum delta | {self.cfg.get('early_stopping_min_delta', 'N/A')} |
| Dropout | {self.cfg.dropout} |
| Hidden layers | {2 if self.cfg.two_layers else 1} |
| Neurons per hidden layer | {self.cfg.l1.n_neurons} |
| Gradient clipping | 1.5 |

## Files

- Metrics: `logs/mlp_snn/version_0/metrics.csv`
- Hyperparameters: `logs/mlp_snn/version_0/hparams.yaml`
- Runtime log: `out.log`
- Checkpoints: `ckpt/`

## Resume command

```powershell
{resume_command}
```

## Evaluation note

This run uses `validate_on: {validation_split}`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
"""

        self._replace_summary(summary_path, summary)
        self._summarized_best_path = best_path


class AppendExperimentWriter(ExperimentWriter):
    """Append resumed metrics instead of deleting an existing CSV log."""

    def _check_log_dir_exists(self):
        # A resumed run intentionally reuses this directory.
        pass

    def __init__(self, log_dir):
        super().__init__(log_dir)
        if self._fs.isfile(self.metrics_file_path):
            with self._fs.open(self.metrics_file_path, "r", newline="") as metrics_file:
                self.metrics_keys = next(csv.reader(metrics_file), [])


class AppendCSVLogger(CSVLogger):
    """CSV logger with a fixed version that is safe to reuse on resume."""

    @property
    def experiment(self):
        if self._experiment is None:
            self._fs.makedirs(self.root_dir, exist_ok=True)
            self._experiment = AppendExperimentWriter(log_dir=self.log_dir)
        return self._experiment


# Main entry point. We use Hydra (https://hydra.cc) for configuration management. Note, that Hydra changes the working directory, such that each run gets a unique directory.

@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: DictConfig):
    logging.getLogger().addHandler(logging.FileHandler("out.log"))
    logging.info(f"Experiment name: {cfg.exp_name}")
    pl.seed_everything(cfg.random_seed, workers=True)
    torch.set_float32_matmul_precision("high")

    datamodule = hydra.utils.instantiate(cfg.dataset)
    model = MLPSNN(cfg)
    callbacks = []
    model_ckpt_tracker: ModelCheckpoint = ModelCheckpoint(
        monitor=cfg.get('tracking_metric', "val_acc_epoch"),
        mode=cfg.get('tracking_mode', 'max'),
        # Keep the latest epoch in addition to the best epoch so interrupted
        # experiments can resume with optimizer and scheduler state intact.
        save_last=True,
        save_top_k=1,
        dirpath="ckpt",
        enable_version_counter=False,
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    callbacks = [
        model_ckpt_tracker,
        lr_monitor,
        GradientNormProgressBar(),
        ApplyLRSchedulerConfig(factor=cfg.factor, patience=cfg.patience),
        TrainingSummaryCallback(model_ckpt_tracker, cfg),
    ]
    if cfg.get("early_stopping", False):
        callbacks.append(
            EarlyStopping(
                monitor=cfg.get("tracking_metric", "val_acc_epoch"),
                mode=cfg.get("tracking_mode", "max"),
                patience=cfg.get("early_stopping_patience", 50),
                min_delta=cfg.get("early_stopping_min_delta", 0.001),
                verbose=True,
            )
        )

    trainer: pl.Trainer = pl.Trainer(
        callbacks=callbacks,
        logger=AppendCSVLogger("logs", name="mlp_snn", version=0),
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.5,
        enable_progress_bar=True,
        strategy=SingleDeviceStrategy(device=cfg.device),
        plugins=[TrustedLocalCheckpointIO()],
        )

    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
    if trainer.interrupted:
        raise RuntimeError("Training was interrupted before all epochs completed.")

    result = trainer.test(model, ckpt_path="best", datamodule=datamodule)
    logging.info(f"Final result: {result}")

    return trainer.checkpoint_callback.best_model_score.cpu().detach().numpy()


if __name__ == "__main__":
    main()
