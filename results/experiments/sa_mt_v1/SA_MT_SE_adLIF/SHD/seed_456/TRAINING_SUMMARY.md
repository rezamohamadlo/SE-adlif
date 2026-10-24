# SHD_SA_MT_SE_adLIF Training Summary — Seed 456

This file is updated automatically whenever training saves a new best checkpoint.

## Run status

- **Status:** Training and testing finished
- **Progress:** 300 epochs completed (`0` through `299`) out of 300
- **Dataset:** SHD
- **Model:** SHD_SA_MT_SE_adLIF
- **Seed:** 456
- **Device:** cuda:0

## Best result

| Metric | Value | Epoch |
|---|---:|---:|
| Best validation accuracy | **99.39%** | 99 |
| Validation loss | **0.0469** | 99 |
| Training accuracy | **100.00%** | 99 |
| Training loss | **0.000144** | 99 |
| Gradient norm | **0.0024** | 99 |
| Learning rate | **0.01** | 99 |

## Model size and computation

| Measurement | Value |
|---|---:|
| Total parameters | **452,920** |
| Trainable parameters | **452,180** |
| Average padded validation timesteps/sample | **293.31** |
| Estimated dense forward GFLOPs/sample | **0.2619** |

The GFLOPs estimate counts dense feed-forward and recurrent matrix multiply-adds, with one multiply-add equal to two FLOPs. It uses the actual average padded validation sequence length. Elementwise neuron-state updates, loss computation, backward propagation, and data loading are excluded. It is therefore a reproducible forward-compute estimate rather than measured GPU throughput.

## Best checkpoint

```text
ckpt/epoch=99-step=2500.ckpt
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
.\.venv\Scripts\python.exe runner.py --models SA_MT_SE_adLIF --datasets SHD --seeds 456 --logdir "results/experiments/sa_mt_v1"
```

## Evaluation note

This run uses `validate_on: 0.2`. If this is the test split, checkpoint selection is not an unbiased final test evaluation.

## Final test evaluation

The best validation checkpoint was evaluated on the test split after training.

| Metric | Value |
|---|---:|
| Test accuracy | **93.86%** |
| Test loss | **0.4139** |

<!-- test-costs:start -->
## Test-set spike activity and inference cost

Validation-selected checkpoint, epoch 99 (zero-based). No training performed.

| Measurement | Value |
|---|---:|
| Total parameters | 452,920.000000 |
| Trainable parameters | 452,180.000000 |
| Hidden spikes/sample | 10,361.562279 |
| L1 spikes/sample | 8,184.950088 |
| L2 spikes/sample | 2,176.612191 |
| L1 firing fraction (%) | 9.088801 |
| L2 firing fraction (%) | 2.416972 |
| Synaptic operations/sample | 9,678,219.876325 |
| Analog input dense MACs/sample | 0.000000 |
| Dense matrix GFLOPs/sample | 0.242895 |
| Matrix + scalar arithmetic GFLOPs/sample | 0.249789 |
| Logical GOp/sample | 0.250380 |
| Forward runtime (ms/sample) | 1.589336 |

Activity uses stored sequence lengths including minimum padding/burn-in; compute includes batch padding.
ECG SynOps exclude analog input. Arithmetic excludes nonlinearities/comparisons; logical operations are not hardware FLOPs or energy.
Runtime is forward-only, excluding data loading, transfer, loss and temporal aggregation.
These test measurements are separate from the validation GFLOPs estimate above.
Detailed measurements, checkpoint identity and environment: [COST_RESULTS.json](COST_RESULTS.json) / [COST_RESULTS.md](COST_RESULTS.md).
<!-- test-costs:end -->
