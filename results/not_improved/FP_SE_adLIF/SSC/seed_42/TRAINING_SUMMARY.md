# SSC_FP_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 40 epochs completed (`0` through `39`) out of 40
- **Dataset:** SSC
- **Model:** SSC_FP_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **79.49%** | 29 |
| Validation loss | **1.3664** | 29 |
| Training accuracy | **85.95%** | 29 |
| Training loss | **0.633312** | 29 |
| Gradient norm | **12.8096** | 29 |
| Learning rate | **0.006** | 29 |

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
ckpt/epoch=29-step=8820.ckpt
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
.\.venv\Scripts\python.exe runner.py --models FP_SE_adLIF --datasets SSC --seeds 42 --logdir "results/valid/protocol_v2"
```

## Evaluation note

SSC uses the official `valid` split for checkpoint selection and the separate official `test` split for final evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **77.67%** |
| Test loss | **1.5382** |

## Evidence audit - 18 October 2026

Unverified training history: metrics.csv contains different validation accuracies at epoch 3/step 1175 (75.55%, 76.28%) and epoch 4/step 1469 (76.05%, 75.37%). The cause is not established. Preserve the recorded 77.67% test score as historical evidence, but exclude it from accepted comparisons until provenance is resolved. No rows were deleted or selected to repair the history.
