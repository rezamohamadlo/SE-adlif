# Paper versus our SE-adLIF reference

Test accuracy (%), **mean ± SD**, for two-layer models. Our results use
seeds **42, 123, 456**; current local statistics use sample SD.

| Dataset / protocol | Paper | Our reference | Difference (pp) | Paper / local runs |
|---|---:|---:|---:|---:|
| SHD: test-based checkpoint selection | 95.81 ± 0.56 | 94.63 ± 0.86 | -1.18 | 20 / 3 |
| SHD*: separate 20% validation split | 93.79 ± 0.76 | 92.93 ± 0.56 | -0.86 | 20 / 3 |
| SSC: official validation/test splits | 80.44 ± 0.26 | 78.21 ± 0.11 | -2.23 | 5 / 3 |
| ECG: training-derived validation | 88.18 ± 0.36 | 88.32 ± 0.35 | +0.14 | 20 / 3 |

SHD* is the appropriate comparison for our current shared protocol. The first
SHD row is historical and uses the test set for checkpoint selection; it is
not an unbiased final evaluation. Its local figures are retained in the Phase 1
report after the original run artifacts were deleted.

Local ECG stopping settings differ: seed 42 disabled early stopping, while
seeds 123 and 456 enabled it. These are descriptive results, not evidence of
a statistically significant improvement or an exact reproduction.

Sources: Table 2, page 11 of `reference_paper.pdf`;
[current reference comparison](paper-vs-reference.md);
[historical Phase 1 results](phase-1-se-adlif-reference.md).

## Seed 42: final test accuracy (%)

| Model | SHD* | SSC | ECG |
|---|---:|---:|---:|
| SE_adLIF reference | 93.42 | **78.33** | 88.13* |
| DTH_SE_adLIF | **93.77** | 78.22 | **88.35** |
| DA_SE_adLIF | 91.83 | 77.88 | 87.20 |
| MR_SE_adLIF (K=2) | 92.58 | 76.93 | 88.05 |
| MT_SE_adLIF | 92.76† | 77.60 | Pending |
| FP_SE_adLIF | 91.56 | 77.67 | 86.85 |

## Seed 42: difference from reference (percentage points)

| Model | SHD* | SSC | ECG |
|---|---:|---:|---:|
| DTH_SE_adLIF | +0.35 | -0.11 | +0.22* |
| DA_SE_adLIF | -1.59 | -0.45 | -0.93* |
| MR_SE_adLIF (K=2) | -0.84 | -1.40 | -0.08* |
| MT_SE_adLIF | -0.66† | -0.73 | Pending |
| FP_SE_adLIF | -1.86 | -0.66 | -1.28* |

*ECG seed 42 reference disabled early stopping; the candidate runs enabled it.
†MT's epoch-104 validation-selected checkpoint was independently evaluated;
its original training logs contain overlapping runs. Pending means final
testing is not yet recorded; — means no result is included.

Source: [seed 42 model comparison](seed-42-model-comparison.md).
