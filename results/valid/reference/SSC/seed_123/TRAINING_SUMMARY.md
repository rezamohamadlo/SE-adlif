# SSC_SE_adLIF Training Summary — Seed 123

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 40 epochs completed (`0` through `39`) out of 40
- **Dataset:** SSC
- **Model:** SSC_SE_adLIF
- **Seed:** 123
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **81.82%** | 13 |
| Validation loss | **0.5963** | 13 |
| Training accuracy | **84.26%** | 13 |
| Training loss | **0.502367** | 13 |
| Gradient norm | **0.3404** | 13 |
| Learning rate | **0.006** | 13 |

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
ckpt/epoch=13-step=4116.ckpt
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
.\.venv\Scripts\python.exe runner.py --models SE_adLIF --datasets SSC --seeds 123 --logdir "results/valid"
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **80.33%** |
| Test loss | **0.6684** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 13 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 1,689,910.000000 |
| Trainable parameters | 1,688,435.000000 |
| Hidden spikes/sample | 43,681.192523 |
| L1 spikes/sample | 18,339.695270 |
| L2 spikes/sample | 25,341.497252 |
| L1 firing fraction (%) | 8.490600 |
| L2 firing fraction (%) | 11.732175 |
| Synaptic operations/sample | 51,577,850.702335 |
| Analog input dense MACs/sample | 0.000000 |
| Dense matrix GFLOPs/sample | 1.008720 |
| Matrix + scalar arithmetic GFLOPs/sample | 1.019584 |
| Logical GOp/sample | 1.020884 |
| Forward runtime (ms/sample) | 1.456407 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
