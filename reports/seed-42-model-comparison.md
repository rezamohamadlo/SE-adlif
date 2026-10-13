# Seed 42: proposed models versus SE_adLIF reference

Final test accuracy (%). MR uses adaptation interval K=2.

| Model | SHD | SSC | ECG |
|---|---:|---:|---:|
| SE_adLIF reference | 93.42 | **78.33** | 88.13* |
| DTH_SE_adLIF | **93.77** | 78.22 | **88.35** |
| DA_SE_adLIF | 91.83 | 77.88 | 87.20 |
| MR_SE_adLIF (K=2) | 92.58 | 76.93 | 88.05 |
| MT_SE_adLIF | 92.76 | 77.60 | Pending |
| FP_SE_adLIF | 91.56 | 77.67 | 86.85 |

Change relative to reference, in percentage points (rounded displayed values):

| Model | SHD | SSC | ECG |
|---|---:|---:|---:|
| DTH_SE_adLIF | +0.35 | -0.11 | +0.22* |
| DA_SE_adLIF | -1.59 | -0.45 | -0.93* |
| MR_SE_adLIF (K=2) | -0.84 | -1.40 | -0.08* |
| MT_SE_adLIF | -0.66 | -0.73 | Pending |
| FP_SE_adLIF | -1.86 | -0.66 | -1.28* |

Numeric scores come from completed final evaluations with separate validation
and test splits. Historical test-selected MT/FP scores are excluded. FP has
completed all three datasets. MT SSC finished with 77.60% test accuracy;
MT ECG has no final test result yet. Differences use the rounded displayed accuracies.
A single seed does not establish statistical significance.

*The ECG reference test score is an independent test measurement, but its
early stopping was disabled while candidate ECG runs enabled it. ECG
differences are descriptive comparisons with unmatched stopping settings.

MT was recovered by evaluating the epoch-104 checkpoint selected by the
latest saved validation callback (98.84% validation accuracy). No retraining
was performed. Its original run folder contained overlapping logs, so the
verified summary and SHA-256 provenance are retained in the main MT folder.
The separate evaluation folder was removed. This recovers one independently evaluated checkpoint;
it does not establish that the original training history was uncontaminated.

## Sources

- SHD reference: `results/valid/reference/SHD/seed_42/hydra/2026-10-12/07-17-03/TRAINING_SUMMARY.md`
- SSC reference: `results/valid/reference/SSC/seed_42/TRAINING_SUMMARY.md`
- ECG reference: `results/valid/reference/ECG/seed_42/TRAINING_SUMMARY.md`
- DTH and DA: `results/valid/protocol_v2/<model>/<dataset>/seed_42/TRAINING_SUMMARY.md`
- MR: `results/valid/protocol_v2/MR_SE_adLIF/K_2/<dataset>/seed_42/TRAINING_SUMMARY.md`
- MT: `results/valid/protocol_v2/MT_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md` and `RESULT.json`
- MT SSC/ECG: `results/valid/protocol_v2/MT_SE_adLIF/<dataset>/seed_42/TRAINING_SUMMARY.md` when available
- FP: `results/valid/protocol_v2/FP_SE_adLIF/<dataset>/seed_42/TRAINING_SUMMARY.md`

The older SHD test-selected reference is excluded; SHD uses the new
training-derived validation run. Scores are taken from final test evaluations.
