# DTH_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| ECG | 123 | test | 87.81 | 88.45 | +0.639 |
| ECG | 123 | validation | 86.29 | 86.36 | +0.072 |
| ECG | 42 | test | 87.81 | 88.21 | +0.400 |
| ECG | 42 | validation | 85.87 | 86.20 | +0.331 |
| ECG | 456 | test | 87.27 | 88.54 | +1.270 |
| ECG | 456 | validation | 85.92 | 86.55 | +0.628 |
| SHD* | 123 | test | 94.88 | 95.23 | +0.353 |
| SHD* | 123 | validation | 99.08 | 99.02 | -0.061 |
| SHD* | 42 | test | 94.70 | 94.52 | -0.177 |
| SHD* | 42 | validation | 99.14 | 99.08 | -0.061 |
| SHD* | 456 | test | 93.90 | 94.17 | +0.265 |
| SHD* | 456 | validation | 99.02 | 98.84 | -0.184 |
| SSC | 42 | test | 80.46 | 80.89 | +0.432 |
| SSC | 42 | validation | 81.84 | 82.37 | +0.531 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| ECG | test | 3 | 88.40 | 0.17 |
| ECG | validation | 3 | 86.37 | 0.18 |
| SHD* | test | 3 | 94.64 | 0.54 |
| SHD* | validation | 3 | 98.98 | 0.13 |
| SSC | test | 1 | 80.89 | unavailable |
| SSC | validation | 1 | 82.37 | unavailable |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
