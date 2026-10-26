# AMT_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| ECG | 123 | test | 87.82 | 88.66 | +0.843 |
| ECG | 123 | validation | 86.60 | 86.53 | -0.064 |
| ECG | 42 | test | 87.65 | 88.41 | +0.759 |
| ECG | 42 | validation | 86.38 | 86.78 | +0.402 |
| ECG | 456 | test | 88.02 | 88.74 | +0.722 |
| ECG | 456 | validation | 86.72 | 87.29 | +0.569 |
| SHD* | 123 | test | 90.64 | 90.55 | -0.088 |
| SHD* | 123 | validation | 99.08 | 99.08 | +0.000 |
| SHD* | 42 | test | 94.48 | 94.74 | +0.265 |
| SHD* | 42 | validation | 98.90 | 98.84 | -0.061 |
| SHD* | 456 | test | 93.02 | 92.98 | -0.044 |
| SHD* | 456 | validation | 99.08 | 99.02 | -0.061 |
| SSC | 123 | test | 80.45 | 81.02 | +0.569 |
| SSC | 123 | validation | 81.95 | 82.42 | +0.471 |
| SSC | 42 | test | 80.24 | 80.68 | +0.446 |
| SSC | 42 | validation | 82.25 | 82.72 | +0.471 |
| SSC | 456 | test | 80.08 | 80.45 | +0.373 |
| SSC | 456 | validation | 81.95 | 82.26 | +0.311 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| ECG | test | 3 | 88.60 | 0.17 |
| ECG | validation | 3 | 86.87 | 0.39 |
| SHD* | test | 3 | 92.76 | 2.11 |
| SHD* | validation | 3 | 98.98 | 0.13 |
| SSC | test | 3 | 80.72 | 0.28 |
| SSC | validation | 3 | 82.46 | 0.23 |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
