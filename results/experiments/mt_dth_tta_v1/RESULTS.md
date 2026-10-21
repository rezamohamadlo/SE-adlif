# MT_DTH_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| SHD* | 123 | test | 93.55 | 93.60 | +0.044 |
| SHD* | 42 | test | 93.82 | 93.73 | -0.088 |
| SHD* | 456 | test | 93.86 | 93.82 | -0.044 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| SHD* | test | 3 | 93.71 | 0.11 |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
