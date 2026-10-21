# MT_SE_adLIF test-time augmentation

Separate exploratory evaluation. Identity plus input-gain views; frozen weights. Validation and test results are separate. Previously observed test scores do not become unseen.

| Dataset | Seed | Split | Baseline (%) | TTA (%) | Difference (pp) |
|---|---:|---|---:|---:|---:|
| SSC | 123 | test | 79.98 | 80.39 | +0.417 |
| SSC | 42 | test | 80.01 | 80.40 | +0.393 |
| SSC | 456 | test | 80.23 | 80.37 | +0.147 |

| Dataset | Split | Completed seeds | TTA mean (%) | Sample SD (pp) |
|---|---|---:|---:|---:|
| SSC | test | 3 | 80.39 | 0.01 |

Three model forwards per batch; not a cost-matched comparison to one-view inference. No energy, latency or accuracy improvement is assumed. ECG accuracy counts labeled timesteps.
