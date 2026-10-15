# SHD_DTH_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training in progress
- **Progress:** 2 epochs completed (`0` through `1`) out of 300
- **Dataset:** SHD
- **Model:** SHD_DTH_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **63.21%** | 1 |
| Validation loss | **1.1202** | 1 |
| Training accuracy | **9.20%** | 1 |
| Training loss | **2.883120** | 1 |
| Gradient norm | **1.1328** | 1 |
| Learning rate | **0.01** | 1 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **451,484** |
| Trainable parameters | **450,744** |
| Average padded validation timesteps/sample | **293.31** |
| Estimated dense forward GFLOPs/sample | **0.2619** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=1-step=50.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 300 |
| Batch size | 256 |
| Initial learning rate | 0.01 |
| LR scheduler | plateau |
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
.\.venv\Scripts\python.exe runner.py --models DTH_SE_adLIF --datasets SHD --seeds 42 --logdir "results/valid/protocol_v2"
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
