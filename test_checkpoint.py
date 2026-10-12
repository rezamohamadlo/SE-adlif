"""Evaluate a trusted local classification checkpoint without training."""
import argparse
import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
import torch

from models.pl_module import MLPSNN

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    cfg = OmegaConf.create(checkpoint['hyper_parameters']['cfg'])
    pl.seed_everything(cfg.random_seed, workers=True)
    torch.set_float32_matmul_precision('high')
    # Standalone execution has no Hydra working directory to resolve data paths.
    data_path = Path(str(cfg.dataset.data_path))
    cfg.dataset.data_path = str(data_path if data_path.is_absolute() else ROOT / data_path)
    cfg.dataset.num_workers = 0
    model = MLPSNN(cfg)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    datamodule = hydra.utils.instantiate(cfg.dataset)
    device = torch.device(str(cfg.device))
    trainer = pl.Trainer(
        accelerator='gpu' if device.type == 'cuda' else 'cpu',
        devices=[device.index or 0] if device.type == 'cuda' else 1,
        logger=False, enable_checkpointing=False,
    )
    results = trainer.test(model, datamodule=datamodule)[0]
    directory = checkpoint_path.parent.parent
    (directory / 'TEST_RESULTS.json').write_text(
        json.dumps(dict(checkpoint=str(checkpoint_path), metrics=results), indent=2) + '\n',
        encoding='utf-8',
    )
    summary_path = directory / 'TRAINING_SUMMARY.md'
    if summary_path.exists():
        marker = '\n## Final test evaluation\n'
        summary = summary_path.read_text(encoding='utf-8').split(marker, 1)[0].rstrip()
        section = (
            f'{marker}\nThe selected validation checkpoint was evaluated without further training.\n\n'
            f'Checkpoint: `{checkpoint_path.name}`\n\n'
            '| Metric | Value |\n|---|---:|\n'
            f"| Test accuracy | **{100 * results['test_acc']:.2f}%** |\n"
            f"| Test loss | **{results['test_loss']:.4f}** |\n"
        )
        summary_path.write_text(summary + '\n' + section, encoding='utf-8')
    print(f"Test accuracy: {100 * results['test_acc']:.2f}%")
    print(f'Results saved in {directory}')
    if (directory / 'SWEEP_CONFIG.json').exists() and (directory.parent / 'LEADERBOARD.md').exists():
        from tune_shd import report
        report(directory.parent)


if __name__ == '__main__':
    main()
