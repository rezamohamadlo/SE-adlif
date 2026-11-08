# SHD_LG_DTH_SE_adLIF Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 300 epochs completed (`0` through `299`) out of 300
- **Dataset:** SHD
- **Model:** SHD_LG_DTH_SE_adLIF
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **92.35%** | 94 |
| Validation loss | **0.3486** | 94 |
| Training accuracy | **99.70%** | 94 |
| Training loss | **0.011635** | 94 |
| Gradient norm | **0.3228** | 94 |
| Learning rate | **0.01** | 94 |

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
ckpt/epoch=94-step=2375.ckpt
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
| Dropout | 0.25 |
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
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 123 --logdir "results/experiments/shd_speaker_augmented_v1"
```

## Evaluation note

This run uses `validate_on: speaker_holdout`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **96.64%** |
| Test loss | **0.1829** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 94 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 452,924.000000 |
| Trainable parameters | 452,184.000000 |
| Hidden spikes/sample | 12,847.044611 |
| L1 spikes/sample | 10,625.749558 |
| L2 spikes/sample | 2,221.295053 |
| L1 firing fraction (%) | 11.799134 |
| L2 firing fraction (%) | 2.466589 |
| Synaptic operations/sample | 11,449,279.010601 |
| Analog input dense MACs/sample | 0.000000 |
| Dense matrix GFLOPs/sample | 0.242895 |
| Matrix + scalar arithmetic GFLOPs/sample | 0.251550 |
| Logical GOp/sample | 0.252338 |
| Forward runtime (ms/sample) | 2.067705 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
