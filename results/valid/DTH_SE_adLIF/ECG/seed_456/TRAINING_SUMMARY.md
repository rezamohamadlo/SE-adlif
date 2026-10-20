# ECG_DTH_SE_adLIF_2layer Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 400 epochs completed (`0` through `399`) out of 400
- **Dataset:** ECG
- **Model:** ECG_DTH_SE_adLIF_2layer
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **85.92%** | 341 |
| Validation loss | **0.3817** | 341 |
| Training accuracy | **87.45%** | 341 |
| Training loss | **0.337027** | 341 |
| Gradient norm | **0.1116** | 341 |
| Learning rate | **0.01** | 341 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **4,768** |
| Trainable parameters | **4,690** |
| Average padded validation timesteps/sample | **1301.00** |
| Estimated dense forward GFLOPs/sample | **0.0111** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=341-step=3078.ckpt
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
.\.venv\Scripts\python.exe runner.py --models DTH_SE_adLIF --datasets ECG --seeds 456 --logdir "results/valid"
```

## Evaluation note

Validation and official test evaluation are distinct; see saved dataset config for the selection protocol.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **87.27%** |
| Test loss | **0.3340** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 341 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 4,768.000000 |
| Trainable parameters | 4,690.000000 |
| Hidden spikes/sample | 31,634.205674 |
| L1 spikes/sample | 11,833.787234 |
| L2 spikes/sample | 19,800.418440 |
| L1 firing fraction (%) | 25.266434 |
| L2 firing fraction (%) | 42.276066 |
| Synaptic operations/sample | 1,682,836.042553 |
| Analog input dense MACs/sample | 187,344.000000 |
| Dense matrix GFLOPs/sample | 0.011053 |
| Matrix + scalar arithmetic GFLOPs/sample | 0.014678 |
| Logical GOp/sample | 0.015150 |
| Forward runtime (ms/sample) | 44.262410 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
