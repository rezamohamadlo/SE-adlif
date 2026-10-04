# SHD_SE_adLIF Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training in progress
- **Progress:** 129 epochs completed (`0` through `128`) out of 300
- **Dataset:** SHD
- **Model:** SHD_SE_adLIF
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **94.70%** | 128 |
| Validation loss | **0.3792** | 128 |
| Training accuracy | **100.00%** | 128 |
| Training loss | **0.000018** | 128 |
| Gradient norm | **0.0003** | 128 |
| Learning rate | **0.006561** | 128 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **450,760** |
| Trainable parameters | **450,020** |
| Average padded validation timesteps/sample | **281.11** |
| Estimated dense forward GFLOPs/sample | **0.2510** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=128-step=1935.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 512 |
| Initial learning rate | 0.01 |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 15 |
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
.\run_phase1_se_adlif.ps1 -Mode reference -Resume 1
```

## Evaluation note

This run uses `validate_on: test`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
