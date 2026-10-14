# Paper versus local SE_adLIF reference

Final test accuracy (%), mean ± SD. Paper values use the two-layer SE-adLIF
rows in Table 2 (page 11 of `reference_paper.pdf`). Local statistics use
seeds 42, 123, and 456 and sample SD (n−1), calculated from final CSV test metrics.

| Dataset | Paper: mean ± SD | Paper runs | Local reference: mean ± sample SD | Local runs | Local − paper mean (pp) |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.79 ± 0.76 | 20 | 93.15 ± 0.42 | 3 | -0.64 |
| SSC | 80.44 ± 0.26 | 5 | 78.21 ± 0.11 | 3 | -2.23 |
| ECG (2 layers) | 88.18 ± 0.36 | 20 | 88.32 ± 0.35 | 3 | +0.14 |

SHD* uses 20% of the official training data for validation and selects the
checkpoint by validation accuracy. This is the appropriate paper row for
the local reference runs. The paper's 95.81 ± 0.56 SHD row instead uses the
test set for checkpoint selection and is excluded from this comparison.
The paper's one-layer ECG result (86.88 ± 0.40) is also excluded because the
local reference has two layers.

The ECG reference seeds used different stopping settings: seed 42 disabled
early stopping; seeds 123 and 456 enabled it. The table describes the retained
results and is not an exact reproduction of every paper training setting.
Three local seeds provide much less evidence about variability than 20 runs;
the small positive ECG mean difference does not demonstrate superiority.

## Local seed results

| Dataset | Seed 42 | Seed 123 | Seed 456 |
|---|---:|---:|---:|
| SHD* | 92.93 | 92.89 | 93.64 |
| SSC | 78.33 | 78.12 | 78.17 |
| ECG | 88.13 | 88.72 | 88.11 |

Local sources: final `test_acc` rows in `metrics.csv` under
`results/valid/reference/<dataset>/seed_<seed>/` (SHD now uses canonical per-seed directories). Values above are rounded after calculating statistics.

Paper: Baronig et al., *Advancing Spatio-Temporal Processing in Spiking Neural
Networks through Adaptation*, Table 2, page 11 in the local PDF;
DOI: 10.1038/s41467-025-60878-z.

## Canonical SHD reference replacement

The current SHD* reference is the completed paper-baseline run set promoted to
`results/valid/reference/SHD/seed_<seed>/`. Its batch is 256 and all 300 epochs
completed without early stopping. The former batch-512/early-stopped SHD
reference is retired. Aggregate metrics and checkpoint provenance are in
`results/valid/reference/SHD/REFERENCE_RESULTS.json` and per-seed
`REFERENCE_RESULT.json`. Proposed controlled-v2 SHD runs have different training
settings and must not be treated as matched comparisons against this baseline.
