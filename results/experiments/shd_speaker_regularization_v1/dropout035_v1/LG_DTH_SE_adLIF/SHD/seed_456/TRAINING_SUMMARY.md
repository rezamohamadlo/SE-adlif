# SHD_LG_DTH_SE_adLIF Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training finished
- **Progress:** 300 epochs completed (`0` through `299`) out of 300
- **Dataset:** SHD
- **Model:** SHD_LG_DTH_SE_adLIF
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **91.25%** | 193 |
| Validation loss | **0.8729** | 193 |
| Training accuracy | **99.55%** | 193 |
| Training loss | **0.017193** | 193 |
| Gradient norm | **0.2498** | 193 |
| Learning rate | **0.01** | 193 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **452,924** |
| Trainable parameters | **452,184** |
| Average padded validation timesteps/sample | **250.00** |
| Estimated dense forward GFLOPs/sample | **0.2232** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=193-step=4850.ckpt
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
| Dropout | 0.35 |
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
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 456 --validation-only --logdir "results/experiments/shd_speaker_regularization_v1/dropout035_v1"
```

## Evaluation note

This run uses `validate_on: speaker_holdout`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
