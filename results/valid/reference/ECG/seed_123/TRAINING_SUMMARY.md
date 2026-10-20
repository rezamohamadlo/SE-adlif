# ECG_SE_adLIF_2layer Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 283 epochs completed (`0` through `282`) out of 400
- **Dataset:** ECG
- **Model:** ECG_SE_adLIF_2layer
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **86.18%** | 232 |
| Validation loss | **0.3765** | 232 |
| Training accuracy | **87.40%** | 232 |
| Training loss | **0.338932** | 232 |
| Gradient norm | **0.1065** | 232 |
| Learning rate | **0.01** | 232 |

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
ckpt/epoch=232-step=2097.ckpt
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

## Runner command

```powershell
.\.venv\Scripts\python.exe runner.py --models SE_adLIF --datasets ECG --seeds 123
```

## Evaluation note

ECG validation is reserved from the training file; the separate test file is used for final evaluation. Check saved stopping and epoch settings before claiming a matched baseline comparison.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **88.72%** |
| Test loss | **0.3045** |

This runner command uses the shared evaluation settings and starts or resumes a runner-managed job. It does not import this retained historical run automatically.

## Evidence audit - 18 October 2026

Historical reference, excluded from matched 400-epoch/no-early-stopping comparisons: early stopping was enabled. The recorded final test value remains descriptive evidence.

## Reporting status - 20 October 2026

Accepted historical ECG reference evidence by user agreement; stopping/budget differs from the 400-epoch proposed-model protocol. Comparisons are descriptive, not fully controlled. Actual completed training: 283 epochs; last checkpoint epoch 282. Accuracy matching the paper does not remove the configuration deviation.
