# Seed 42: proposed models versus SE_adLIF reference

Final test accuracy (%). MR uses adaptation interval K=2.

| Model | SHD | SSC | ECG |
|---|---:|---:|---:|
| SE_adLIF reference | 93.42 | **78.33** | 88.13* |
| DTH_SE_adLIF | **93.77** | 78.22 | **88.35** |
| DA_SE_adLIF | 91.83 | 77.88 | 87.20 |
| MR_SE_adLIF (K=2) | 92.58 | 76.93 | 88.05 |
| MT_SE_adLIF | 93.73† | — | — |
| FP_SE_adLIF | 94.26† | — | — |

Change relative to reference, in percentage points (rounded displayed values):

| Model | SHD | SSC | ECG |
|---|---:|---:|---:|
| DTH_SE_adLIF | +0.35 | -0.11 | +0.22* |
| DA_SE_adLIF | -1.59 | -0.45 | -0.93* |
| MR_SE_adLIF (K=2) | -0.84 | -1.40 | -0.08* |
| MT_SE_adLIF | +0.31† | — | — |
| FP_SE_adLIF | +0.84† | — | — |

*The saved ECG reference has early stopping disabled, while the controlled-v2
candidate runs have it enabled. Its summary reports 383 completed epochs.
The ECG comparison is descriptive; rerun the reference using the shared
protocol for a fully matched comparison. A single seed does not establish
statistical significance.

†MT and FP used `validate_on: test`: the official test set was used for
checkpoint selection and final evaluation. These are historical results,
not matched controlled-protocol comparisons; their differences from the
reference are descriptive only. MT was manually stopped after validation
through epoch 156, with early stopping disabled; checkpoint epoch 104 was
evaluated. FP finished with early stopping enabled, selecting epoch 62.
Bold SHD values identify the best result among the controlled runs only.
An em dash means no result is included for that dataset.

## Sources

- SHD reference: `results/reference_shd/seed_42/hydra/2026-10-12/07-17-03/TRAINING_SUMMARY.md`
- SSC reference: `results/phase1_models/reference/SSC/seed_42/TRAINING_SUMMARY.md`
- ECG reference: `results/phase1_models/reference/ECG/seed_42/TRAINING_SUMMARY.md`
- DTH and DA: `results/full_runs/protocol_v2/<model>/<dataset>/seed_42/TRAINING_SUMMARY.md`
- MR: `results/full_runs/protocol_v2/MR_SE_adLIF/K_2/<dataset>/seed_42/TRAINING_SUMMARY.md`
- MT: `results/model_development/ours/MT_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md`
- FP: `results/model_development/ours/FP_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md`

The older SHD test-selected reference is excluded; SHD uses the new
training-derived validation run. Scores are taken from final test evaluations.
