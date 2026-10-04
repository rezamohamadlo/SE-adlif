import os
import tempfile

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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
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
        save_last=False,
        save_top_k=1,
        dirpath="ckpt"
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )
    callbacks = [model_ckpt_tracker, lr_monitor]

    trainer: pl.Trainer = pl.Trainer(
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger("logs", name="mlp_snn"),
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.5,
        enable_progress_bar=True,
        strategy=SingleDeviceStrategy(device=cfg.device),
        plugins=[TrustedLocalCheckpointIO()],
        )

    trainer.fit(model, datamodule=datamodule)
    result = trainer.test(model, ckpt_path="best", datamodule=datamodule)
    logging.info(f"Final result: {result}")

    return trainer.checkpoint_callback.best_model_score.cpu().detach().numpy()


if __name__ == "__main__":
    main()
