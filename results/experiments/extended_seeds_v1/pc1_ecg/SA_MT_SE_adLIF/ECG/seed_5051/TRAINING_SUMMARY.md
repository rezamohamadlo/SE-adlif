# ECG_SA_MT_SE_adLIF_2layer Training Summary — Seed 5051

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training finished
- **Progress:** 400 epochs completed (`0` through `399`) out of 400
- **Dataset:** ECG
- **Model:** ECG_SA_MT_SE_adLIF_2layer
- **Seed:** 5051
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **86.23%** | 388 |
| Validation loss | **0.3681** | 388 |
| Training accuracy | **87.77%** | 388 |
| Training loss | **0.324969** | 388 |
| Gradient norm | **0.0352** | 388 |
| Learning rate | **0.01** | 388 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **4,908** |
| Trainable parameters | **4,830** |
| Average padded validation timesteps/sample | **1301.00** |
| Estimated dense forward GFLOPs/sample | **0.0111** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=388-step=3501.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 400 |
| Batch size | 64 |
| Initial learning rate | 0.01 |
| LR scheduler | plateau |
| LR scheduler factor | 0.9 |
| LR scheduler patience | 9999 |
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
.\.venv\Scripts\python.exe runner.py --models SA_MT_SE_adLIF --datasets ECG --seeds 5051 --validation-only --seed-cohort extension_v1 --logdir "results/experiments/extended_seeds_v1/pc1_ecg"
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
