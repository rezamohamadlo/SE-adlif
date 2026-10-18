# LG_DTH_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: paper_classification_v1. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.77% | 93.02% | 94.43% | 3 | 93.74% ± 0.71% |
| SSC | pending | pending | pending | 0 | pending |
| ECG | pending | pending | pending | 0 | pending |

## Seed-456 completion and provenance

The fresh protocol-corrected run completed 300 epochs with batch 256 and early stopping disabled. Final test accuracy is 94.43463087081909%, loss 0.41735318303108215, recorded in `SHD/seed_456/logs/mlp_snn/version_0/metrics.csv`. The validation-selected checkpoint is `epoch=226-step=5675.ckpt` (99.02% validation accuracy). The last checkpoint reaches epoch 299, step 7500.

The previous early-stopped seed-456 run (93.51% test accuracy) is retained at `archive/lg_dth_seed456_early_stopping_20261017/` and excluded from the current aggregate. This fresh rerun followed observation of that earlier test score; it is a protocol correction, not selection between repeated runs by test accuracy. See `SHD/seed_456/RUN_RELOCATION.json` for its relocation record; original state backups remain at `archive/lg_dth_seed456_relocation_20261017/`.

## Spike activity and inference cost

| Measurement | SHD* value | Evidence / scope |
|---|---:|---|
| Total parameters | 452,924 | Per-seed training summaries; identical across all three seeds |
| Trainable parameters | 452,184 | Per-seed training summaries; identical across all three seeds |
| Estimated dense forward GFLOPs/sample | 0.2619 | Validation; average padded length 293.31 timesteps; matrix operations only |
| Estimated dense forward GFLOPs/test sample | 0.2429 | Actual batch-padded official-test lengths; matrix operations only |
| Total hidden spikes/sample | 10,723 ± 669 | Mean ± sample SD across three seeds; definitions in linked comparison |
| Spikes/sample, L1 / L2 | 8,652 ± 469 / 2,071 ± 381 | Mean ± sample SD across three seeds; definitions in linked comparison |
| Firing fraction (%), L1 / L2 | 9.61 ± 0.52 / 2.30 ± 0.42 | Mean ± sample SD across three seeds; definitions in linked comparison |
| Estimated synaptic operations/sample | 9,973,248 ± 396,815 | Mean ± sample SD across three seeds; definitions in linked comparison |
| Non-matrix logical operations/sample, padded | 9,442,372 ± 0 | Mean ± sample SD across three seeds; definitions in linked comparison |
| Inference time (ms/sample) | 1.901 ± 0.026 | Mean ± sample SD across three seeds; definitions in linked comparison |

All three seeds measured on the official test set. Activity uses pre-batch sequence lengths, including configured minimum padding; non-matrix operations and runtime include batch padding. See the [measurement protocol](../../../../reports/reference-vs-models.md#spike-activity-and-inference-cost).
