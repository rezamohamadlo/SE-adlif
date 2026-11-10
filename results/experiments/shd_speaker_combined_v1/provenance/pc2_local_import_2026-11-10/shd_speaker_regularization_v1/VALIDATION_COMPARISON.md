# SHD speaker holdout: regularization tuning

Validation accuracy (%) only. Official test evaluation is deferred.
Missing or unfinished baseline runs remain pending; they do not block new training.
Split seed 42; validation speakers [0, 3]; fingerprint `e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
Fresh training: 300 epochs, batch 256, baseline LR/schedule; full resume only for interrupted runs.

| Model | Variant | Seed 42 | Seed 123 | Seed 456 | n | Mean +/- sample SD |
|---|---|---:|---:|---:|---:|---:|
| DTH_SE_adLIF | augmented_v1 | 90.0306 | 90.0306 | 91.6208 | 3 | 90.5607 +/- 0.9181 |
| DTH_SE_adLIF | dropout030_v1 | pending | 89.6024 | 90.8868 | 2 | 90.2446 +/- 0.9082 |
| DTH_SE_adLIF | dropout035_v1 | pending | 92.1713 | pending | 1 | 92.1713 (SD unavailable) |
| DTH_SE_adLIF | spike_delete010_v1 | pending | pending | pending | 0 | pending |
| LG_DTH_SE_adLIF | augmented_v1 | 91.1927 | 92.3547 | 91.7431 | 3 | 91.7635 +/- 0.5813 |
| LG_DTH_SE_adLIF | dropout030_v1 | pending | 91.7431 | 92.2936 | 2 | 92.0183 +/- 0.3892 |
| LG_DTH_SE_adLIF | dropout035_v1 | pending | 90.5810 | pending | 1 | 90.5810 (SD unavailable) |
| LG_DTH_SE_adLIF | spike_delete010_v1 | pending | pending | pending | 0 | pending |
