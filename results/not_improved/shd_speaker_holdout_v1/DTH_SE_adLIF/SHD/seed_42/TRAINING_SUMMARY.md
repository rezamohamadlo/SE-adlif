# SHD_DTH_SE_adLIF Training Summary — Seed 42

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 300 epochs completed (`0` through `299`) out of 300
- **Dataset:** SHD
- **Model:** SHD_DTH_SE_adLIF
- **Seed:** 42
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **87.71%** | 170 |
| Validation loss | **0.7307** | 170 |
| Training accuracy | **99.95%** | 170 |
| Training loss | **0.001927** | 170 |
| Gradient norm | **0.0479** | 170 |
| Learning rate | **0.01** | 170 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **451,484** |
| Trainable parameters | **450,744** |
| Average padded validation timesteps/sample | **250.00** |
| Estimated dense forward GFLOPs/sample | **0.2232** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=170-step=4275.ckpt
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
.\.venv\Scripts\python.exe runner.py --shd-speaker-phase 1
```

## Evaluation note

This run uses `validate_on: speaker_holdout`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **92.58%** |
| Test loss | **0.4382** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 170 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 451,484.000000 |
| Trainable parameters | 450,744.000000 |
| Hidden spikes/sample | 10,529.966431 |
| L1 spikes/sample | 8,866.090989 |
| L2 spikes/sample | 1,663.875442 |
| L1 firing fraction (%) | 9.845159 |
| L2 firing fraction (%) | 1.847615 |
| Synaptic operations/sample | 9,972,646.466431 |
| Analog input dense MACs/sample | 0.000000 |
| Dense matrix GFLOPs/sample | 0.242895 |
| Matrix + scalar arithmetic GFLOPs/sample | 0.250372 |
| Logical GOp/sample | 0.251353 |
| Forward runtime (ms/sample) | 2.626326 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
