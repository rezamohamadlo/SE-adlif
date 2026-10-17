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

The previous early-stopped seed-456 run (93.51% test accuracy) is retained at `results/archive/lg_dth_seed456_early_stopping_20261017/` and excluded from the current aggregate. This fresh rerun followed observation of that earlier test score; it is a protocol correction, not selection between repeated runs by test accuracy. See `SHD/seed_456/RUN_RELOCATION.json` for its relocation record; original state backups remain at `results/archive/lg_dth_seed456_relocation_20261017/`.

## Spike activity and inference cost

| Measurement | SHD* value | Evidence / scope |
|---|---:|---|
| Total parameters | 452,924 | Per-seed training summaries; identical across all three seeds |
| Trainable parameters | 452,184 | Per-seed training summaries; identical across all three seeds |
| Estimated dense forward GFLOPs/sample | 0.2619 | Validation; average padded length 293.31 timesteps; matrix operations only |
| Total hidden spikes/sample | Pending | Sum of L1 and L2 spikes per sample |
| Spikes/sample, L1 / L2 | Pending | Requires inference instrumentation of validation-selected checkpoints |
| Firing fraction (%), L1 / L2 | Pending | Requires spike counts and valid neuron-timestep denominators |
| Estimated synaptic operations/sample | Pending | Requires presynaptic activity, fan-out and recurrent-use accounting |
| Neuron-state operations/sample | Pending | Variant-specific elementwise updates under a documented operation-count convention |
| Inference time (ms/sample) | Pending | Requires a separate benchmark on matching hardware and batching |

These pending measurements were not recorded during training. The dense estimate excludes elementwise neuron updates and does not measure energy. See the [seven-model comparison and measurement protocol](../../../../reports/reference-vs-models.md#spike-activity-and-inference-cost). All three seeds completed; spike and timing measurements remain pending a separate inference evaluation.
