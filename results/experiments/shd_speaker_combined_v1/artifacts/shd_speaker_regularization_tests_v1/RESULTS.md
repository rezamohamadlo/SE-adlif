# SHD speaker-holdout regularization: exploratory ordinary test results

Exploratory official-test evaluation explicitly requested before full three-seed hyperparameter selection. Every checkpoint was selected by validation accuracy; these test scores must not select settings.

Frozen weights, one ordinary forward per batch, no TTA or training.
Original validation-only run directories and selection evidence are preserved.

| Dropout | Model | Seed | Selected epoch (zero-based) | Best validation (%) | Test (%) |
|---:|---|---:|---:|---:|---:|
| 0.30 | DTH_SE_adLIF | 123 | 205 | 89.6024 | 91.2986 |
| 0.30 | LG_DTH_SE_adLIF | 123 | 296 | 91.7431 | 93.2420 |
| 0.30 | DTH_SE_adLIF | 456 | 276 | 90.8868 | 94.2580 |
| 0.30 | LG_DTH_SE_adLIF | 456 | 228 | 92.2936 | 95.0972 |
| 0.35 | DTH_SE_adLIF | 123 | 145 | 92.1713 | 95.8481 |
| 0.35 | LG_DTH_SE_adLIF | 123 | 106 | 90.5810 | 93.8163 |
