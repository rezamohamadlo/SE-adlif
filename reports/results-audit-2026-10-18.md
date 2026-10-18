# Local results evidence audit - 18 October 2026

Scope: current records in `results/valid/` and `results/not_improved/`. Archived results are preserved as historical evidence. Canonical metrics are read from each seed's `logs/mlp_snn/version_0/metrics.csv`; nested historical Hydra logs are not pooled into the active run.

Checked per-seed best validation and final test values against summaries, model tables and general reports; inspected saved configs for budget, batch and stopping differences. Existing SHD* and reference aggregate accuracy values agree with canonical metrics. Paper-source and cost values were not remeasured or rederived. No training, checkpoint changes or raw-metric edits were performed.

| Seed directory | Final test (%) | Validation rows | Evidence status |
|---|---:|---:|---|
| [results/valid/DTH_SE_adLIF/SHD/seed_123](../results/valid/DTH_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) | 94.88 | 300 | No metric/config discrepancy found |
| [results/valid/DTH_SE_adLIF/SHD/seed_42](../results/valid/DTH_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) | 94.70 | 300 | No metric/config discrepancy found |
| [results/valid/DTH_SE_adLIF/SHD/seed_456](../results/valid/DTH_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) | 93.90 | 300 | No metric/config discrepancy found |
| [results/valid/DTH_SE_adLIF/SSC/seed_123](../results/valid/DTH_SE_adLIF/SSC/seed_123/TRAINING_SUMMARY.md) | 78.31 | 40 | Checkpoints missing locally |
| [results/valid/DTH_SE_adLIF/SSC/seed_42](../results/valid/DTH_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) | 78.22 | 40 | Historical metadata; checked shared settings match |
| [results/valid/DTH_SE_adLIF/SSC/seed_456](../results/valid/DTH_SE_adLIF/SSC/seed_456/TRAINING_SUMMARY.md) | 78.04 | 40 | Checkpoints missing locally |
| [results/valid/MT_SE_adLIF/SHD/seed_123](../results/valid/MT_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) | 94.52 | 300 | No metric/config discrepancy found |
| [results/valid/MT_SE_adLIF/SHD/seed_42](../results/valid/MT_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) | 94.88 | 300 | No metric/config discrepancy found |
| [results/valid/MT_SE_adLIF/SHD/seed_456](../results/valid/MT_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) | 94.43 | 300 | No metric/config discrepancy found |
| [results/valid/MT_SE_adLIF/SSC/seed_123](../results/valid/MT_SE_adLIF/SSC/seed_123/TRAINING_SUMMARY.md) | pending | unavailable | Partial summary only; no local metrics/checkpoint |
| [results/valid/MT_SE_adLIF/SSC/seed_42](../results/valid/MT_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) | 77.60 | 40 | Historical metadata; checked shared settings match |
| [results/valid/reference/ECG/seed_123](../results/valid/reference/ECG/seed_123/TRAINING_SUMMARY.md) | 88.72 | 283 | Historical unmatched budget/stopping |
| [results/valid/reference/ECG/seed_42](../results/valid/reference/ECG/seed_42/TRAINING_SUMMARY.md) | 88.13 | 383 | Historical unmatched budget/stopping |
| [results/valid/reference/ECG/seed_456](../results/valid/reference/ECG/seed_456/TRAINING_SUMMARY.md) | 88.11 | 185 | Historical unmatched budget/stopping |
| [results/valid/reference/SHD/seed_123](../results/valid/reference/SHD/seed_123/TRAINING_SUMMARY.md) | 92.89 | 300 | No metric/config discrepancy found |
| [results/valid/reference/SHD/seed_42](../results/valid/reference/SHD/seed_42/TRAINING_SUMMARY.md) | 92.93 | 300 | No metric/config discrepancy found |
| [results/valid/reference/SHD/seed_456](../results/valid/reference/SHD/seed_456/TRAINING_SUMMARY.md) | 93.64 | 300 | No metric/config discrepancy found |
| [results/valid/reference/SSC/seed_123](../results/valid/reference/SSC/seed_123/TRAINING_SUMMARY.md) | 78.12 | 40 | Historical metadata; checked shared settings match |
| [results/valid/reference/SSC/seed_42](../results/valid/reference/SSC/seed_42/TRAINING_SUMMARY.md) | 78.33 | 40 | Historical metadata; checked shared settings match |
| [results/valid/reference/SSC/seed_456](../results/valid/reference/SSC/seed_456/TRAINING_SUMMARY.md) | 78.17 | 40 | Historical metadata; checked shared settings match |
| [results/not_improved/DA_SE_adLIF/ECG/seed_42](../results/not_improved/DA_SE_adLIF/ECG/seed_42/TRAINING_SUMMARY.md) | 87.20 | 329 | Historical unmatched budget/stopping |
| [results/not_improved/DA_SE_adLIF/SHD/seed_123](../results/not_improved/DA_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) | 93.99 | 300 | No metric/config discrepancy found |
| [results/not_improved/DA_SE_adLIF/SHD/seed_42](../results/not_improved/DA_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) | 93.73 | 300 | No metric/config discrepancy found |
| [results/not_improved/DA_SE_adLIF/SHD/seed_456](../results/not_improved/DA_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) | 92.27 | 300 | No metric/config discrepancy found |
| [results/not_improved/DA_SE_adLIF/SSC/seed_42](../results/not_improved/DA_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) | 77.88 | 40 | Historical metadata; checked shared settings match |
| [results/not_improved/FP_SE_adLIF/ECG/seed_42](../results/not_improved/FP_SE_adLIF/ECG/seed_42/TRAINING_SUMMARY.md) | 86.85 | 204 | Historical unmatched budget/stopping |
| [results/not_improved/FP_SE_adLIF/SHD/seed_123](../results/not_improved/FP_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) | 92.49 | 300 | No metric/config discrepancy found |
| [results/not_improved/FP_SE_adLIF/SHD/seed_42](../results/not_improved/FP_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) | 93.51 | 300 | No metric/config discrepancy found |
| [results/not_improved/FP_SE_adLIF/SHD/seed_456](../results/not_improved/FP_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) | 94.04 | 300 | No metric/config discrepancy found |
| [results/not_improved/FP_SE_adLIF/SSC/seed_42](../results/not_improved/FP_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) | 77.67 | 42 | Conflicting history; exclude pending provenance resolution; Historical metadata; checked shared settings match |
| [results/not_improved/LG_DTH_SE_adLIF/SHD/seed_123](../results/not_improved/LG_DTH_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) | 93.02 | 300 | No metric/config discrepancy found |
| [results/not_improved/LG_DTH_SE_adLIF/SHD/seed_42](../results/not_improved/LG_DTH_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) | 93.77 | 300 | No metric/config discrepancy found |
| [results/not_improved/LG_DTH_SE_adLIF/SHD/seed_456](../results/not_improved/LG_DTH_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) | 94.43 | 300 | No metric/config discrepancy found |
| [results/not_improved/MR_SE_adLIF/K_2/ECG/seed_42](../results/not_improved/MR_SE_adLIF/K_2/ECG/seed_42/TRAINING_SUMMARY.md) | 88.05 | 201 | Historical unmatched budget/stopping |
| [results/not_improved/MR_SE_adLIF/K_2/SHD/seed_123](../results/not_improved/MR_SE_adLIF/K_2/SHD/seed_123/TRAINING_SUMMARY.md) | 93.95 | 300 | No metric/config discrepancy found |
| [results/not_improved/MR_SE_adLIF/K_2/SHD/seed_42](../results/not_improved/MR_SE_adLIF/K_2/SHD/seed_42/TRAINING_SUMMARY.md) | 94.04 | 300 | No metric/config discrepancy found |
| [results/not_improved/MR_SE_adLIF/K_2/SHD/seed_456](../results/not_improved/MR_SE_adLIF/K_2/SHD/seed_456/TRAINING_SUMMARY.md) | 93.68 | 300 | No metric/config discrepancy found |
| [results/not_improved/MR_SE_adLIF/K_2/SSC/seed_42](../results/not_improved/MR_SE_adLIF/K_2/SSC/seed_42/TRAINING_SUMMARY.md) | 76.93 | 40 | Historical metadata; checked shared settings match |
| [results/not_improved/MT_DTH_SE_adLIF/SHD/seed_123](../results/not_improved/MT_DTH_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) | 93.55 | 300 | No metric/config discrepancy found |
| [results/not_improved/MT_DTH_SE_adLIF/SHD/seed_42](../results/not_improved/MT_DTH_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) | 93.82 | 300 | No metric/config discrepancy found |
| [results/not_improved/MT_DTH_SE_adLIF/SHD/seed_456](../results/not_improved/MT_DTH_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) | 93.86 | 300 | No metric/config discrepancy found |

Missing checkpoints limit independent verification even when metrics and saved configs agree. Completion here means a final test row exists, not that every original artifact is available. No live process status was established.

The three SHD* reference seeds also contain nested historical Hydra metrics; these are separate histories, not duplicate epochs in their canonical CSVs. FP SSC seed 42 differs: its canonical CSV itself contains conflicting rows at identical epochs and steps. Its cause is unresolved, so the score is excluded from accepted comparisons without deleting evidence.

Per-seed notes correct the generic SSC/ECG validation warning. Model reports distinguish historical early-stopped ECG results and incomplete/unverified SSC records. `results/valid/README.md` now describes the current runner and paths.
