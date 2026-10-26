# ECG_MT_TC_SE_adLIF_2layer Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 400 epochs completed (`0` through `399`) out of 400
- **Dataset:** ECG
- **Model:** ECG_MT_TC_SE_adLIF_2layer
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **85.75%** | 379 |
| Validation loss | **0.3895** | 379 |
| Training accuracy | **87.46%** | 379 |
| Training loss | **0.334174** | 379 |
| Gradient norm | **0.0866** | 379 |
| Learning rate | **0.01** | 379 |

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
ckpt/epoch=379-step=3420.ckpt
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
.\.venv\Scripts\python.exe runner.py --models MT_TC_SE_adLIF --datasets ECG --seeds 456 --logdir "results/experiments/mt_consistency_runner_v1"
```

## Evaluation note

This run uses `validate_on: validation`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **88.62%** |
| Test loss | **0.3034** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 379 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 4,836.000000 |
| Trainable parameters | 4,758.000000 |
| Hidden spikes/sample | 27,201.730496 |
| L1 spikes/sample | 10,802.609929 |
| L2 spikes/sample | 16,399.120567 |
| L1 firing fraction (%) | 23.064758 |
| L2 firing fraction (%) | 35.013922 |
| Synaptic operations/sample | 1,465,860.595745 |
| Analog input dense MACs/sample | 187,344.000000 |
| Dense matrix GFLOPs/sample | 0.011053 |
| Matrix + scalar arithmetic GFLOPs/sample | 0.014033 |
| Logical GOp/sample | 0.014322 |
| Forward runtime (ms/sample) | 46.726616 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
