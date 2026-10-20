# ECG_MT_SE_adLIF_2layer Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 400 epochs completed (`0` through `399`) out of 400
- **Dataset:** ECG
- **Model:** ECG_MT_SE_adLIF_2layer
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **86.25%** | 398 |
| Validation loss | **0.3725** | 398 |
| Training accuracy | **87.53%** | 398 |
| Training loss | **0.330747** | 398 |
| Gradient norm | **0.0915** | 398 |
| Learning rate | **0.01** | 398 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **4,836** |
| Trainable parameters | **4,758** |
| Average padded validation timesteps/sample | **1301.00** |
| Estimated dense forward GFLOPs/sample | **0.0111** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=398-step=3591.ckpt
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
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets ECG --seeds 42 --logdir "results/valid"
```

## Evaluation note

Validation and official test evaluation are distinct; see saved dataset config for the selection protocol.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **87.26%** |
| Test loss | **0.3428** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 398 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 4,836.000000 |
| Trainable parameters | 4,758.000000 |
| Hidden spikes/sample | 27,022.510638 |
| L1 spikes/sample | 11,078.198582 |
| L2 spikes/sample | 15,944.312057 |
| L1 firing fraction (%) | 23.653170 |
| L2 firing fraction (%) | 34.042856 |
| Synaptic operations/sample | 1,466,596.680851 |
| Analog input dense MACs/sample | 187,344.000000 |
| Dense matrix GFLOPs/sample | 0.011053 |
| Matrix + scalar arithmetic GFLOPs/sample | 0.014033 |
| Logical GOp/sample | 0.014322 |
| Forward runtime (ms/sample) | 45.256762 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
