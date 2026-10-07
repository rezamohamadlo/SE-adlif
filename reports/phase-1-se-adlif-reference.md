# Phase 1 Report: SE-adLIF Reference Baselines

**Date:** 2026-10-07  
**Status:** Phase 1 reference-baseline acquisition completed  
**Model:** SE-adLIF  
**Datasets:** SHD, SSC, and ECG/QTDB  
**Seeds:** 42, 123, and 456

## Objective

Phase 1 established SE-adLIF reference results on three temporal-classification
datasets before development of the proposed model. The canonical result set
contains exactly one run for each dataset and seed combination. Accuracy
statistics below are the arithmetic mean and sample standard deviation across
the three seeds.

## Final results

| Dataset | Seed 42 | Seed 123 | Seed 456 | Mean ± sample SD | Reported metric |
|---|---:|---:|---:|---:|---|
| SHD | 95.45% | 94.70% | 93.73% | **94.63% ± 0.86%** | Test-split accuracy used during checkpoint selection |
| SSC | 78.33% | 78.12% | 78.17% | **78.21% ± 0.11%** | Final test accuracy |
| ECG/QTDB | 88.13% | 88.72% | 88.11% | **88.32% ± 0.35%** | Final test accuracy |

SSC and ECG used validation data for checkpoint selection followed by a final
evaluation of the selected checkpoint on the test split. SHD used the test
split as validation, matching the current reference setup; its result is useful
for reproduction but is not an unbiased final-test estimate.

## Per-seed checkpoint results

### SHD

| Seed | Selected epoch | Accuracy | Loss | Checkpoint |
|---:|---:|---:|---:|---|
| 42 | 205 | 95.45% | 0.2864 | `seed_42/ckpt/epoch=205-step=3090.ckpt` |
| 123 | 128 | 94.70% | 0.3792 | See canonical seed directory |
| 456 | 60 | 93.73% | 0.3182 | See canonical seed directory |

The SHD seed-42 `TRAINING_SUMMARY.md` contains an older 94.96% snapshot from
epoch 108. The authoritative `metrics.csv` records the later maximum of
95.4505% at epoch 205, and the corresponding checkpoint is present.

### SSC

| Seed | Best validation accuracy | Selected epoch | Final test accuracy | Final test loss |
|---:|---:|---:|---:|---:|
| 42 | 80.29% | 18 | 78.33% | 0.9854 |
| 123 | 79.78% | 25 | 78.12% | 1.1006 |
| 456 | 79.49% | 19 | 78.17% | 1.0292 |

### ECG/QTDB

| Seed | Best validation accuracy | Selected epoch | Final test accuracy | Final test loss |
|---:|---:|---:|---:|---:|
| 42 | 86.34% | 267 | 88.13% | 0.3119 |
| 123 | 86.18% | 232 | 88.72% | 0.3045 |
| 456 | 84.77% | 168 | 88.11% | 0.3186 |

## Architecture and training configuration

| Setting | SHD | SSC | ECG/QTDB |
|---|---:|---:|---:|
| Hidden layers | 2 | 2 | 2 |
| Neurons per hidden layer | 360 | 720 | 36 |
| Total parameters | 450,760 | 1,689,910 | 4,692 |
| Trainable parameters | 450,020 | 1,688,435 | 4,614 |
| Batch size in saved runs | 512 | 256 | 64 |
| Epoch limit | 300 | 40 | 400 |
| Initial learning rate | 0.01 | 0.006 | 0.01 |
| Dropout | 0.15 | 0.15 | 0.15 |
| SLAYER alpha | 5.0 | 5.0 | 5.0 |
| SLAYER c | 0.4 | 0.4 | 0.2 |
| `tau_u` hidden range | [5, 25] | [5, 25] | [5, 25] |
| `tau_w` hidden range | [60, 300] | [60, 300] | [60, 300] |
| Reparameterization q | 120 | 120 | 120 |
| Output `tau_u` | 15 | 15 | 3 |
| Loss aggregation | `sum_t softmax(y_t)` | `softmax(sum_t y_t)` via cross-entropy | `softmax(sum_t y_t)` via cross-entropy |

For SHD and SSC, the first 10 valid timesteps are ignored. Padded timesteps are
excluded by the sequence mask. The legacy SHD mode named `softmax` and the
explicit mode `sum_softmax_over_time` both compute the same operation:

```text
sum_t softmax(y_t)
```

The corrected SSC configuration uses `summed_membrane_potentials`, which sums
the output membrane-potential logits over time before `CrossEntropyLoss` applies
log-softmax. Earlier SSC runs that used a temporal mean are excluded from this
report.

## Model size and estimated computation

| Dataset | Average padded validation timesteps/sample | Estimated dense forward GFLOPs/sample |
|---|---:|---:|
| SHD | 281.11 | 0.2510 |
| SSC | 300.00 | 1.0087 |
| ECG/QTDB | 1301.00 | 0.0111 |

The GFLOPs estimate counts dense feed-forward and recurrent matrix
multiply-adds, treating one multiply-add as two FLOPs. It excludes elementwise
neuron-state updates, loss computation, backward propagation, and data loading.

## Canonical artifacts

```text
results/phase1_models/reference/SHD/seed_42
results/phase1_models/reference/SHD/seed_123
results/phase1_models/reference/SHD/seed_456

results/phase1_models/reference/SSC/seed_42
results/phase1_models/reference/SSC/seed_123
results/phase1_models/reference/SSC/seed_456

results/phase1_models/reference/ECG/seed_42
results/phase1_models/reference/ECG/seed_123
results/phase1_models/reference/ECG/seed_456
```

Each canonical directory contains its available training summary, metrics,
hyperparameters, runtime log, and checkpoints. Large runtime artifacts may be
excluded from Git by `.gitignore`; the local result directories remain the
authoritative experimental record.

## Scientific limitations

1. SHD uses the test split for checkpoint selection. A final publication-grade
   comparison should introduce a separate validation split and evaluate the
   chosen checkpoint on the test split only once.
2. The SHD runs were not fully uniform: seed 42 used an effectively disabled LR
   scheduler, while seeds 123 and 456 used scheduler patience 15 and early
   stopping. The results remain the accepted Phase 1 reference set, but a
   controlled comparison with a proposed method should standardize this
   protocol.
3. ECG seed 42 was manually finalized after 383 epochs, whereas seeds 123 and
   456 stopped earlier with patience-50 early stopping. All reported ECG test
   values use each seed's best validation checkpoint.
4. Because only three seeds were evaluated, uncertainty estimates are
   descriptive and should be interpreted cautiously.

## Phase 1 conclusion

The SE-adLIF reference baseline is now available on all three target datasets.
The next phase can implement the proposed model and compare it against these
canonical runs using the same data preprocessing, seeds, evaluation splits,
and reporting procedure. For the final paper, the standardized protocol should
resolve the SHD test-selection issue and keep stopping and scheduling settings
identical across seeds and models.
