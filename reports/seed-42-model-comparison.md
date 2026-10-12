# Seed 42: proposed models versus SE_adLIF reference

Final test accuracy (%). MR uses adaptation interval K=2.

| Model | SHD | SSC | ECG |
|---|---:|---:|---:|
| SE_adLIF reference | 93.42 | **78.33** | — |
| DTH_SE_adLIF | **93.77** | 78.22 | **88.35** |
| DA_SE_adLIF | 91.83 | 77.88 | 87.20 |
| MR_SE_adLIF (K=2) | 92.58 | 76.93 | 88.05 |

Change relative to reference, in percentage points (rounded displayed values):

| Model | SHD | SSC | ECG |
|---|---:|---:|---:|
| DTH_SE_adLIF | +0.35 | -0.11 | — |
| DA_SE_adLIF | -1.59 | -0.45 | — |
| MR_SE_adLIF (K=2) | -0.84 | -1.40 | — |

Only completed runs with matched validation/training settings are included.
Historical test-selected MT/FP scores and the ECG reference with early
stopping disabled are archived and excluded. FP SSC is being resumed;
new completed results can be added after final testing. An em dash means
no matching completed result is available. A single seed does not establish
statistical significance.

## Sources

- SHD reference: `results/valid/reference/SHD/seed_42/hydra/2026-10-12/07-17-03/TRAINING_SUMMARY.md`
- SSC reference: `results/valid/reference/SSC/seed_42/TRAINING_SUMMARY.md`
- DTH and DA: `results/valid/protocol_v2/<model>/<dataset>/seed_42/TRAINING_SUMMARY.md`
- MR: `results/valid/protocol_v2/MR_SE_adLIF/K_2/<dataset>/seed_42/TRAINING_SUMMARY.md`

The older SHD test-selected reference is excluded; SHD uses the new
training-derived validation run. Scores are taken from final test evaluations.
