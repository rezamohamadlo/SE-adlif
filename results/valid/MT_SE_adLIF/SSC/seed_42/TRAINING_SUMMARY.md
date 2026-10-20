# SSC_MT_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 40 epochs completed (`0` through `39`) out of 40
- **Dataset:** SSC
- **Model:** SSC_MT_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **82.00%** | 23 |
| Validation loss | **0.6121** | 23 |
| Training accuracy | **86.09%** | 23 |
| Training loss | **0.439392** | 23 |
| Gradient norm | **0.2190** | 23 |
| Learning rate | **0.006** | 23 |

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
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SSC --seeds 42 --logdir "results/valid"
```

## Evaluation note

SSC uses official validation for checkpoint selection and the separate official test set for final evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **80.01%** |
| Test loss | **0.7035** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 23 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 1,692,790.000000 |
| Trainable parameters | 1,691,315.000000 |
| Hidden spikes/sample | 48,897.197380 |
| L1 spikes/sample | 22,131.506722 |
| L2 spikes/sample | 26,765.690658 |
| L1 firing fraction (%) | 10.246068 |
| L2 firing fraction (%) | 12.391523 |
| Synaptic operations/sample | 58,091,668.516583 |
| Analog input dense MACs/sample | 0.000000 |
| Dense matrix GFLOPs/sample | 1.008720 |
| Matrix + scalar arithmetic GFLOPs/sample | 1.022190 |
| Logical GOp/sample | 1.023493 |
| Forward runtime (ms/sample) | 1.562921 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
