# ECG_SE_adLIF_2layer Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 186 epochs completed (`0` through `185`) out of 400
- **Dataset:** ECG
- **Model:** ECG_SE_adLIF_2layer
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **84.77%** | 168 |
| Validation loss | **0.4003** | 168 |
| Training accuracy | **86.18%** | 168 |
| Training loss | **0.373298** | 168 |
| Gradient norm | **0.1313** | 168 |
| Learning rate | **0.01** | 168 |

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
ckpt/epoch=168-step=1521.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 400 |
| Batch size | 64 |
| Initial learning rate | 0.01 |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
| Early stopping | true |
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
.\run_phase1_se_adlif.ps1 -Mode reference -Datasets ECG -Seeds 456 -Epochs 400 -BatchSize 64 -EarlyStopping 1 -LrSchedulerPatience 9999 -LrSchedulerFactor 0.9 -Resume 1 -LogDir results/phase1_models
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **88.11%** |
| Test loss | **0.3186** |
