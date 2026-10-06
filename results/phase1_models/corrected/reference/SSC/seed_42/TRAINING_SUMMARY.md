# SSC_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 40 epochs completed (`0` through `39`) out of 40
- **Dataset:** SSC
- **Model:** SSC_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **80.29%** | 18 |
| Validation loss | **0.9095** | 18 |
| Training accuracy | **82.29%** | 18 |
| Training loss | **0.685440** | 18 |
| Gradient norm | **14.0330** | 18 |
| Learning rate | **0.006** | 18 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **1,689,910** |
| Trainable parameters | **1,688,435** |
| Average padded validation timesteps/sample | **300.00** |
| Estimated dense forward GFLOPs/sample | **1.0087** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=18-step=5586.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 40 |
| Batch size | 256 |
| Initial learning rate | 0.006 |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
| Early stopping | false |
| Early-stopping patience | 50 |
| Early-stopping minimum delta | 0.001 |
| Dropout | 0.15 |
| Hidden layers | 2 |
| Neurons per hidden layer | 720 |
| Gradient clipping | 1.5 |

## Files

- Metrics: `logs/mlp_snn/version_0/metrics.csv`
- Hyperparameters: `logs/mlp_snn/version_0/hparams.yaml`
- Runtime log: `out.log`
- Checkpoints: `ckpt/`

## Resume command

```powershell
.\run_phase1_se_adlif.ps1 -Mode reference -Datasets SSC -Seeds 42 -Epochs 40 -BatchSize 256 -EarlyStopping 0 -LrSchedulerPatience 9999 -LrSchedulerFactor 0.9 -Resume 1 -LogDir results/phase1_models/corrected
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **78.33%** |
| Test loss | **0.9854** |
