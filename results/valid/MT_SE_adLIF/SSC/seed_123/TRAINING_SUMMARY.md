# SSC_MT_SE_adLIF Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training interrupted; resumable
- **Progress:** 36 epochs completed (`0` through `35`) out of 40
- **Dataset:** SSC
- **Model:** SSC_MT_SE_adLIF
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **80.07%** | 20 |
| Validation loss | **0.9861** | 20 |
| Training accuracy | **83.40%** | 20 |
| Training loss | **0.661569** | 20 |
| Gradient norm | **11.2096** | 20 |
| Learning rate | **0.006** | 20 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **1,692,790** |
| Trainable parameters | **1,691,315** |
| Average padded validation timesteps/sample | **300.00** |
| Estimated dense forward GFLOPs/sample | **1.0087** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=20-step=6174.ckpt
```

## Main hyperparameters

| Hyperparameter | Value |
|---|---:|
| Epoch limit | 40 |
| Batch size | 256 |
| Initial learning rate | 0.006 |
| LR scheduler | plateau |
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
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SSC --seeds 123 --logdir "results/valid"
```

## Evaluation note

SSC uses the official `valid` split for checkpoint selection and the separate official `test` split for final evaluation.

## Evidence audit - 18 October 2026

This is a partial progress snapshot through epoch 20, not a verified live process status. No local metrics CSV, checkpoint or final test result is available. Do not count this seed as completed or infer test accuracy from validation. The saved config identifies `paper_classification_v1`.
