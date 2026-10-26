# MT_TC_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| ECG | 123 | test | 87.98 | 88.68 | +0.694 |
| ECG | 123 | validation | 86.47 | 86.94 | +0.466 |
| ECG | 42 | test | 87.78 | 88.46 | +0.677 |
| ECG | 42 | validation | 86.08 | 86.17 | +0.095 |
| ECG | 456 | test | 88.62 | 89.25 | +0.629 |
| ECG | 456 | validation | 85.75 | 86.48 | +0.725 |
| SHD* | 123 | test | 92.84 | 92.84 | +0.000 |
| SHD* | 123 | validation | 99.20 | 99.08 | -0.123 |
| SHD* | 42 | test | 94.74 | 94.66 | -0.088 |
| SHD* | 42 | validation | 99.08 | 98.84 | -0.245 |
| SHD* | 456 | test | 93.82 | 93.90 | +0.088 |
| SHD* | 456 | validation | 99.26 | 99.08 | -0.184 |
| SSC | 123 | test | 80.05 | 80.47 | +0.427 |
| SSC | 123 | validation | 82.23 | 82.82 | +0.591 |
| SSC | 42 | test | 80.12 | 80.37 | +0.250 |
| SSC | 42 | validation | 82.21 | 82.40 | +0.190 |
| SSC | 456 | test | 80.26 | 80.56 | +0.299 |
| SSC | 456 | validation | 81.91 | 82.28 | +0.371 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| ECG | test | 3 | 88.80 | 0.41 |
| ECG | validation | 3 | 86.53 | 0.39 |
| SHD* | test | 3 | 93.80 | 0.91 |
| SHD* | validation | 3 | 99.00 | 0.14 |
| SSC | test | 3 | 80.47 | 0.10 |
| SSC | validation | 3 | 82.50 | 0.28 |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
