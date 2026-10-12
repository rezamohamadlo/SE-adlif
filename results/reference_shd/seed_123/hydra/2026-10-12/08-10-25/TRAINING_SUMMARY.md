# SHD_SE_adLIF Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 121 epochs completed (`0` through `120`) out of 300
- **Dataset:** SHD
- **Model:** SHD_SE_adLIF
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **98.34%** | 69 |
| Validation loss | **0.0926** | 69 |
| Training accuracy | **100.00%** | 69 |
| Training loss | **0.000107** | 69 |
| Gradient norm | **0.0056** | 69 |
| Learning rate | **0.01** | 69 |

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
ckpt/epoch=69-step=840.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 512 |
| Initial learning rate | 0.01 |
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
.\run_phase1_se_adlif.ps1 -Mode reference -Datasets SHD -Seeds 123 -Epochs 300 -BatchSize 512 -EarlyStopping 1 -LrSchedulerPatience 9999 -LrSchedulerFactor 0.9 -Resume 1 -LogDir results/phase1_models
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **92.31%** |
| Test loss | **0.5552** |
