# MT_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| ECG | 123 | test | 88.19 | 88.95 | +0.761 |
| ECG | 123 | validation | 86.31 | 86.72 | +0.415 |
| ECG | 42 | test | 87.26 | 87.92 | +0.655 |
| ECG | 42 | validation | 86.25 | 86.91 | +0.666 |
| ECG | 456 | test | 87.98 | 88.74 | +0.757 |
| ECG | 456 | validation | 86.17 | 86.46 | +0.297 |
| SHD* | 123 | validation | 99.20 | 99.08 | -0.123 |
| SHD* | 42 | validation | 99.26 | 99.26 | +0.000 |
| SHD* | 456 | validation | 99.33 | 99.08 | -0.245 |
| SSC | 42 | test | 80.01 | 80.40 | +0.393 |
| SSC | 42 | validation | 82.00 | 82.41 | +0.411 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| ECG | test | 3 | 88.53 | 0.54 |
| ECG | validation | 3 | 86.70 | 0.23 |
| SHD* | validation | 3 | 99.14 | 0.11 |
| SSC | test | 1 | 80.40 | unavailable |
| SSC | validation | 1 | 82.41 | unavailable |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
