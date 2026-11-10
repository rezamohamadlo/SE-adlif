# SHD speaker holdout: regularization tuning

Validation accuracy (%) only. Official test evaluation is deferred.
Missing or unfinished baseline runs remain pending; they do not block new training.
Split seed 42; validation speakers [0, 3]; fingerprint `e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
Fresh training: 300 epochs, batch 256, baseline LR/schedule; full resume only for interrupted runs.

| Model | Variant | Seed 42 | Seed 123 | Seed 456 | n | Mean +/- sample SD |
|---|---|---:|---:|---:|---:|---:|
| DTH_SE_adLIF | augmented_v1 | 90.0306 | 90.0306 | 91.6208 | 3 | 90.5607 +/- 0.9181 |
| DTH_SE_adLIF | dropout030_v1 | 90.8868 | 89.6024 | 90.8868 | 3 | 90.4587 +/- 0.7416 |
| DTH_SE_adLIF | dropout035_v1 | 90.5199 | 92.1713 | 90.4587 | 3 | 91.0499 +/- 0.9716 |
| DTH_SE_adLIF | spike_delete010_v1 | 90.6422 | 91.3761 | 92.5382 | 3 | 91.5189 +/- 0.9560 |
| LG_DTH_SE_adLIF | augmented_v1 | 91.1927 | 92.3547 | 91.7431 | 3 | 91.7635 +/- 0.5813 |
| LG_DTH_SE_adLIF | dropout030_v1 | 91.0092 | 91.7431 | 92.2936 | 3 | 91.6820 +/- 0.6444 |
| LG_DTH_SE_adLIF | dropout035_v1 | 91.0092 | 90.5810 | 91.2538 | 3 | 90.9480 +/- 0.3405 |
| LG_DTH_SE_adLIF | spike_delete010_v1 | 90.0917 | 90.4587 | 92.3547 | 3 | 90.9684 +/- 1.2145 |
