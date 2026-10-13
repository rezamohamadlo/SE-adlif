# SHD_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 268 epochs completed (`0` through `267`) out of 300
- **Dataset:** SHD
- **Model:** SHD_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **93.01%** | 216 |
| Validation loss | **0.2490** | 216 |
| Training accuracy | **97.14%** | 216 |
| Training loss | **0.090999** | 216 |
| Gradient norm | **0.7550** | 216 |
| Learning rate | **7.7863212e-05** | 216 |

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
ckpt/epoch=216-step=2604.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 512 |
| Initial learning rate | 0.0003 |
| LR scheduler | cosine |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
| Early stopping | true |
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
.\.ven\Scripts\python.exe tune_shd.py --checkpoint "C:\Users\jafari.h.SPADANACO\Desktop\SE-adlif\results\valid\reference\SHD\seed_42\hydra\2026-10-12\07-17-03\ckpt\epoch=162-step=1956.ckpt" --output "C:\Users\jafari.h.SPADANACO\Desktop\SE-adlif\results\experiments\shd_lr_sweep" --trial c08 --full-train --seeds 42
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **87.50%** |
| Test loss | **0.4646** |
