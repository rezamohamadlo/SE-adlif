# LT_DTH_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| ECG | 123 | test | 87.11 | 87.76 | +0.649 |
| ECG | 123 | validation | 86.75 | 87.30 | +0.551 |
| ECG | 42 | test | 88.33 | 88.96 | +0.636 |
| ECG | 42 | validation | 86.51 | 86.73 | +0.215 |
| ECG | 456 | test | 87.58 | 88.52 | +0.944 |
| ECG | 456 | validation | 86.14 | 86.83 | +0.689 |
| SHD* | 123 | test | 93.42 | 93.42 | +0.000 |
| SHD* | 123 | validation | 98.96 | 98.71 | -0.245 |
| SHD* | 42 | test | 94.66 | 94.74 | +0.088 |
| SHD* | 42 | validation | 98.96 | 98.84 | -0.123 |
| SHD* | 456 | test | 93.86 | 93.64 | -0.221 |
| SHD* | 456 | validation | 99.51 | 99.33 | -0.184 |
| SSC | 123 | test | 80.25 | 80.48 | +0.226 |
| SSC | 123 | validation | 81.54 | 81.69 | +0.140 |
| SSC | 42 | test | 79.95 | 80.37 | +0.427 |
| SSC | 42 | validation | 81.65 | 81.81 | +0.160 |
| SSC | 456 | test | 80.31 | 80.95 | +0.638 |
| SSC | 456 | validation | 81.59 | 82.05 | +0.461 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| ECG | test | 3 | 88.42 | 0.61 |
| ECG | validation | 3 | 86.95 | 0.31 |
| SHD* | test | 3 | 93.93 | 0.71 |
| SHD* | validation | 3 | 98.96 | 0.32 |
| SSC | test | 3 | 80.60 | 0.31 |
| SSC | validation | 3 | 81.85 | 0.18 |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
