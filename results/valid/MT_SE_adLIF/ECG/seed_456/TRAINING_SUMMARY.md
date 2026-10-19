# ECG_MT_SE_adLIF_2layer Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 400 epochs completed (`0` through `399`) out of 400
- **Dataset:** ECG
- **Model:** ECG_MT_SE_adLIF_2layer
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **86.17%** | 393 |
| Validation loss | **0.3770** | 393 |
| Training accuracy | **87.34%** | 393 |
| Training loss | **0.337737** | 393 |
| Gradient norm | **0.0628** | 393 |
| Learning rate | **0.01** | 393 |

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
ckpt/epoch=393-step=3546.ckpt
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
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets ECG --seeds 456 --logdir "results/valid"
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **87.98%** |
| Test loss | **0.3212** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 393 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 4,836.000000 |
| Trainable parameters | 4,758.000000 |
| Hidden spikes/sample | 26,269.503546 |
| L1 spikes/sample | 10,114.226950 |
| L2 spikes/sample | 16,155.276596 |
| L1 firing fraction (%) | 21.594985 |
| L2 firing fraction (%) | 34.493288 |
| Synaptic operations/sample | 1,406,039.234043 |
| Analog input dense MACs/sample | 187,344.000000 |
| Dense matrix GFLOPs/sample | 0.011053 |
| Matrix + scalar arithmetic GFLOPs/sample | 0.014033 |
| Logical GOp/sample | 0.014322 |
| Forward runtime (ms/sample) | 44.230404 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
