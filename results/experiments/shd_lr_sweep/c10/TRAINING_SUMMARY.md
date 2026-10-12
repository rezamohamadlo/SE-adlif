# SHD_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training finished
- **Progress:** 10 epochs completed (`0` through `9`) out of 10
- **Dataset:** SHD
- **Model:** SHD_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **98.90%** | 3 |
| Validation loss | **0.0919** | 3 |
| Training accuracy | **99.97%** | 3 |
| Training loss | **0.001438** | 3 |
| Gradient norm | **0.0003** | 3 |
| Learning rate | **0.001** | 3 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **450,760** |
| Trainable parameters | **450,020** |
| Average padded validation timesteps/sample | **293.31** |
| Estimated dense forward GFLOPs/sample | **0.2619** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=3-step=100.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 10 |
| Batch size | 256 |
| Initial learning rate | 0.001 |
| LR scheduler | constant |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
| Early stopping | false |
| Early-stopping patience | 50 |
| Early-stopping minimum delta | 0.001 |
| Dropout | 0.15 |
| Hidden layers | 2 |
| Neurons per hidden layer | 360 |
| Gradient clipping | 1.5 |

## Files

- Metrics: `logs/mlp_snn/version_0/metrics.csv`
- Hyperparameters: `logs/mlp_snn/version_0/hparams.yaml`
- Runtime log: `out.log`
- Checkpoints: `ckpt/`

## Resume command

```powershell
.\.ven\Scripts\python.exe tune_shd.py --checkpoint "C:\Users\jafari.h.SPADANACO\Desktop\SE-adlif\results\valid\reference\SHD\seed_42\hydra\2026-10-12\07-17-03\ckpt\epoch=162-step=1956.ckpt" --output "C:\Users\jafari.h.SPADANACO\Desktop\SE-adlif\results\experiments\shd_lr_sweep" --trial c10 --epochs 10
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The selected validation checkpoint was evaluated without further training.

Checkpoint: `epoch=3-step=100.ckpt`

| Metric | Value |
|---|---:|
| Test accuracy | **93.02%** |
| Test loss | **0.5971** |
