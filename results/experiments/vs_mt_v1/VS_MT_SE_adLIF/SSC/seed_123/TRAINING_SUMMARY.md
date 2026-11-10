# SSC_VS_MT_SE_adLIF Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training interrupted; resumable
- **Progress:** 30 epochs completed (`0` through `29`) out of 40
- **Dataset:** SSC
- **Model:** SSC_VS_MT_SE_adLIF
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **81.55%** | 23 |
| Validation loss | **0.6322** | 23 |
| Training accuracy | **86.27%** | 23 |
| Training loss | **0.436060** | 23 |
| Gradient norm | **0.1626** | 23 |
| Learning rate | **0.006** | 23 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **1,694,230** |
| Trainable parameters | **1,692,755** |
| Average padded validation timesteps/sample | **300.00** |
| Estimated dense forward GFLOPs/sample | **1.0087** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=23-step=7056.ckpt
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
.\.venv\Scripts\python.exe runner.py --models VS_MT_SE_adLIF --datasets SSC --seeds 123 --logdir "results/experiments/vs_mt_v1"
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.
