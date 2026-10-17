# LG-DTH-SE-adLIF results

Current SHD* baseline suite: seeds 42/123 completed; fresh seed 456 is training after full resume in `SHD/seed_456/`. Early stopping is disabled for all current baseline runs.

| Dataset | Seed 42 | Seed 123 | Seed 456 | Completed matched | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.77% | 93.02% | pending | 2 | 93.40% ± 0.53% |
| SSC | pending | pending | pending | 0 | pending |
| ECG | pending | pending | pending | 0 | pending |

| Run | State / provenance |
|---|---|
| Fresh baseline seed 456 | In progress: 252/300 epochs completed (latest validation epoch 251, 98.96%); best validation 99.02% at epoch 226. Final test pending. |
| Progress snapshot | 2026-10-17 10:56 Tehran; based on completed validation rows in metrics, not the best-checkpoint summary. |
| Relocation | Moved from `results/experiments/lg_dth_shd_seed456_fresh_baseline/` to the canonical seed directory. Weights, optimizer, scheduler, epoch/loop state and checkpoint-selection scores verified unchanged. See [relocation record](SHD/seed_456/RUN_RELOCATION.json). |
| Original state backups | `results/archive/lg_dth_seed456_relocation_20261017/`; original configs and checkpoints retained before location-only metadata edits. |
| Prior early-stopping seed 456 | 93.51% test accuracy, 105 reported completed epochs. Preserved at [archived run](../../../archive/lg_dth_seed456_early_stopping_20261017/). Excluded from matched statistics. |
| Historical mixed-stopping aggregate | 93.43% ± 0.38% across seeds 42/123 and the old early-stopped seed 456; descriptive only. |

Shared settings: `paper_classification_v1`, SHD section of `config/paper_baseline.yaml`; batch 256, LR 0.01, maximum 300 epochs, dropout 0.15, gradient clipping 1.5, scheduler patience 9999/factor 0.9. Validation uses 20% of official training data with split seed 42. Checkpoints are selected by validation accuracy before official test evaluation.

The fresh seed-456 run is a protocol-corrected rerun after the old test score was seen. Preserve this disclosure and do not choose between repeated seed-456 runs using test accuracy. No pending test score is inferred from validation accuracy.

## Spike activity and inference cost

| Measurement | SHD* value | Evidence / scope |
|---|---:|---|
| Total parameters | 452,924 | Per-seed training summaries; identical across all three seeds |
| Trainable parameters | 452,184 | Per-seed training summaries; identical across all three seeds |
| Estimated dense forward GFLOPs/sample | 0.2619 | Validation; average padded length 293.31 timesteps; matrix operations only |
| Spikes/sample, L1 / L2 | Pending | Requires inference instrumentation of validation-selected checkpoints |
| Firing fraction (%), L1 / L2 | Pending | Requires spike counts and valid neuron-timestep denominators |
| Estimated synaptic operations/sample | Pending | Requires presynaptic activity, fan-out and recurrent-use accounting |
| Inference time (ms/sample) | Pending | Requires a separate benchmark on matching hardware and batching |

These pending measurements were not recorded during training. The dense estimate excludes elementwise neuron updates and does not measure energy. See the [seven-model comparison and measurement protocol](../../../../reports/reference-vs-models.md#spike-activity-and-inference-cost). Seed 456 remains incomplete; architecture counts do not imply a completed final-test result.
