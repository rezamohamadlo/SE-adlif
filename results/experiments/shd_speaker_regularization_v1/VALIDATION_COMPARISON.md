# SHD speaker holdout: regularization tuning

Validation accuracy (%) only. Official test evaluation is deferred.
Missing or unfinished baseline runs remain pending; they do not block new training.
Split seed 42; validation speakers [0, 3]; fingerprint `e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
Fresh training: 300 epochs, batch 256, baseline LR/schedule; full resume only for interrupted runs.

| Model | Variant | Seed 42 | Seed 123 | Seed 456 | n | Mean +/- sample SD |
|---|---|---:|---:|---:|---:|---:|
| DTH_SE_adLIF | augmented_v1 | pending | pending | pending | 0 | pending |
| DTH_SE_adLIF | dropout030_v1 | 90.8868 | pending | pending | 1 | 90.8868 (SD unavailable) |
| DTH_SE_adLIF | dropout035_v1 | 90.5199 | pending | pending | 1 | 90.5199 (SD unavailable) |
| DTH_SE_adLIF | spike_delete010_v1 | 90.6422 | pending | pending | 1 | 90.6422 (SD unavailable) |
| LG_DTH_SE_adLIF | augmented_v1 | pending | pending | pending | 0 | pending |
| LG_DTH_SE_adLIF | dropout030_v1 | 91.0092 | pending | pending | 1 | 91.0092 (SD unavailable) |
| LG_DTH_SE_adLIF | dropout035_v1 | 91.0092 | pending | pending | 1 | 91.0092 (SD unavailable) |
| LG_DTH_SE_adLIF | spike_delete010_v1 | 90.0917 | pending | pending | 1 | 90.0917 (SD unavailable) |
