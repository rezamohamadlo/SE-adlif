# Paper versus local SE_adLIF reference

Final test accuracy (%), mean ± SD. Paper values use the two-layer SE-adLIF
rows in Table 2 (page 11 of `reference_paper.pdf`). Local statistics use
seeds 42, 123, and 456 and sample SD (n−1), calculated from final CSV test metrics.

| Dataset | Paper: mean ± SD | Paper runs | Local reference: mean ± sample SD | Local runs | Local − paper mean (pp) |
|---|---:|---:|---:|---:|---:|
| SHD* | 93.79 ± 0.76 | 20 | 92.93 ± 0.56 | 3 | -0.86 |
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
| SHD | 93.42 | 92.31 | 93.07 |
| SSC | 78.33 | 78.12 | 78.17 |
| ECG | 88.13 | 88.72 | 88.11 |

Local sources: final `test_acc` rows in `metrics.csv` under
`results/valid/reference/<dataset>/seed_<seed>/` (SHD retains its Hydra date/time
subdirectories). Values above are rounded after calculating statistics.

Paper: Baronig et al., *Advancing Spatio-Temporal Processing in Spiking Neural
Networks through Adaptation*, Table 2, page 11 in the local PDF;
DOI: 10.1038/s41467-025-60878-z.
