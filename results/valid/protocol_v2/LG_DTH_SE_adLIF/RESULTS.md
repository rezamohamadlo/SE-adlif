# LG-DTH-SE-adLIF results

Current SHD* baseline suite: seeds 42/123 completed; fresh seed 456 is interrupted and ready for full resume in `SHD/seed_456/`. Early stopping is disabled for all current baseline runs.

| Dataset | Seed 42 | Seed 123 | Seed 456 | Completed matched | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.77% | 93.02% | pending | 2 | 93.40% ± 0.53% |
| SSC | pending | pending | pending | 0 | pending |
| ECG | pending | pending | pending | 0 | pending |

| Run | State / provenance |
|---|---|
| Fresh baseline seed 456 | Random initialization and fresh optimizer at epoch 0; last saved checkpoint is epoch 115 (116 saved epochs), step 2900. Continue toward 300 total epochs with early stopping disabled. |
| Latest interrupted summary | Reports 117 epochs of progress; resume restores `last.ckpt`, so work after the last saved state may be replayed. |
| Relocation | Moved from `results/experiments/lg_dth_shd_seed456_fresh_baseline/` to the canonical seed directory. Weights, optimizer, scheduler, epoch/loop state and checkpoint-selection scores verified unchanged. See [relocation record](SHD/seed_456/RUN_RELOCATION.json). |
| Original state backups | `results/archive/lg_dth_seed456_relocation_20261017/`; original configs and checkpoints retained before location-only metadata edits. |
| Prior early-stopping seed 456 | 93.51% test accuracy, 105 reported completed epochs. Preserved at [archived run](../../../archive/lg_dth_seed456_early_stopping_20261017/). Excluded from matched statistics. |
| Historical mixed-stopping aggregate | 93.43% ± 0.38% across seeds 42/123 and the old early-stopped seed 456; descriptive only. |

Shared settings: `paper_classification_v1`, SHD section of `config/paper_baseline.yaml`; batch 256, LR 0.01, maximum 300 epochs, dropout 0.15, gradient clipping 1.5, scheduler patience 9999/factor 0.9. Validation uses 20% of official training data with split seed 42. Checkpoints are selected by validation accuracy before official test evaluation.

The fresh seed-456 run is a protocol-corrected rerun after the old test score was seen. Preserve this disclosure and do not choose between repeated seed-456 runs using test accuracy. No pending test score is inferred from validation accuracy.
