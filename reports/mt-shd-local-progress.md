Local training-progress snapshot preserved when syncing origin/main. The verified final result remains in results/valid/protocol_v2/MT_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md.

# SHD_MT_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training in progress
- **Progress:** 55 epochs completed (`0` through `54`) out of 300
- **Dataset:** SHD
- **Model:** SHD_MT_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **98.34%** | 54 |
| Validation loss | **0.0807** | 54 |
| Training accuracy | **99.79%** | 54 |
| Training loss | **0.005839** | 54 |
| Gradient norm | **0.0611** | 54 |
| Learning rate | **0.01** | 54 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **452,200** |
| Trainable parameters | **451,460** |
| Average padded validation timesteps/sample | **296.92** |
| Estimated dense forward GFLOPs/sample | **0.2651** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=54-step=660.ckpt
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
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SHD --seeds 42 --logdir "results/valid/protocol_v2"
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
