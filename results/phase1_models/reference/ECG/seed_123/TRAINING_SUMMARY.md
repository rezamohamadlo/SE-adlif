# ECG_SE_adLIF_2layer Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training interrupted; resumable
- **Progress:** 132 epochs completed (`0` through `131`) out of 300
- **Dataset:** ECG
- **Model:** ECG_SE_adLIF_2layer
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **33.78%** | 130 |
| Validation loss | **1.5122** | 130 |
| Training accuracy | **34.63%** | 130 |
| Training loss | **1.483527** | 130 |
| Gradient norm | **0.5421** | 130 |
| Learning rate | **0.00531441** | 130 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **4,692** |
| Trainable parameters | **4,614** |
| Average padded validation timesteps/sample | **1301.00** |
| Estimated dense forward GFLOPs/sample | **0.0111** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=130-step=131.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 512 |
| Initial learning rate | 0.01 |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 15 |
| Early stopping | false |
| Early-stopping patience | 50 |
| Early-stopping minimum delta | 0.001 |
| Dropout | 0.15 |
| Hidden layers | 2 |
| Neurons per hidden layer | 36 |
| Gradient clipping | 1.5 |

## Files

- Metrics: `logs/mlp_snn/version_0/metrics.csv`
- Hyperparameters: `logs/mlp_snn/version_0/hparams.yaml`
- Runtime log: `out.log`
- Checkpoints: `ckpt/`

## Resume command

```powershell
.\run_phase1_se_adlif.ps1 -Mode reference -Datasets ECG -Seeds 123 -Epochs 300 -BatchSize 512 -EarlyStopping 0 -LrSchedulerPatience 15 -LrSchedulerFactor 0.9 -Resume 1 -LogDir results/phase1_models
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
