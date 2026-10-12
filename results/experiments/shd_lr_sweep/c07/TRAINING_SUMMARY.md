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
| Best validation accuracy | **98.96%** | 0 |
| Validation loss | **0.0798** | 0 |
| Training accuracy | **N/A** | 0 |
| Training loss | **N/A** | 0 |
| Gradient norm | **0.0001** | 0 |
| Learning rate | **0.0003** | 0 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **450,760** |
| Trainable parameters | **450,020** |
| Average padded validation timesteps/sample | **296.92** |
| Estimated dense forward GFLOPs/sample | **0.2651** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=0-step=12.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 10 |
| Batch size | 512 |
| Initial learning rate | 0.0003 |
| LR scheduler | plateau |
| LR scheduler factor | 0.5 |
| LR scheduler patience | 2 |
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
.\.ven\Scripts\python.exe tune_shd.py --checkpoint "C:\Users\jafari.h.SPADANACO\Desktop\SE-adlif\results\valid\reference\SHD\seed_42\hydra\2026-10-12\07-17-03\ckpt\epoch=162-step=1956.ckpt" --output "C:\Users\jafari.h.SPADANACO\Desktop\SE-adlif\results\experiments\shd_lr_sweep" --trial c07 --epochs 10
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The selected validation checkpoint was evaluated without further training.

Checkpoint: `epoch=0-step=12.ckpt`

| Metric | Value |
|---|---:|
| Test accuracy | **93.60%** |
| Test loss | **0.5927** |
