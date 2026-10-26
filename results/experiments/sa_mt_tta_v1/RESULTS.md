# SA_MT_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| ECG | 123 | test | 88.65 | 89.31 | +0.660 |
| ECG | 123 | validation | 86.90 | 87.22 | +0.313 |
| ECG | 42 | test | 88.26 | 88.75 | +0.496 |
| ECG | 42 | validation | 86.03 | 86.25 | +0.215 |
| ECG | 456 | test | 88.27 | 89.11 | +0.839 |
| ECG | 456 | validation | 86.33 | 86.63 | +0.297 |
| SHD* | 123 | test | 93.15 | 93.11 | -0.044 |
| SHD* | 123 | validation | 99.14 | 99.02 | -0.123 |
| SHD* | 42 | test | 94.17 | 94.30 | +0.133 |
| SHD* | 42 | validation | 98.90 | 98.77 | -0.123 |
| SHD* | 456 | test | 93.86 | 93.90 | +0.044 |
| SHD* | 456 | validation | 99.39 | 99.26 | -0.123 |
| SSC | 123 | test | 80.32 | 80.79 | +0.466 |
| SSC | 123 | validation | 81.95 | 82.41 | +0.461 |
| SSC | 42 | test | 80.13 | 80.42 | +0.285 |
| SSC | 42 | validation | 81.82 | 82.07 | +0.250 |
| SSC | 456 | test | 79.84 | 80.40 | +0.564 |
| SSC | 456 | validation | 81.89 | 81.95 | +0.060 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| ECG | test | 3 | 89.06 | 0.28 |
| ECG | validation | 3 | 86.70 | 0.49 |
| SHD* | test | 3 | 93.77 | 0.61 |
| SHD* | validation | 3 | 99.02 | 0.25 |
| SSC | test | 3 | 80.54 | 0.22 |
| SSC | validation | 3 | 82.14 | 0.24 |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
