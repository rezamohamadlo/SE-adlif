# SHD speaker holdout: regularization tuning

Validation accuracy only; test results are exploratory and remain in separate evaluation directories.
Combined saved completion records from PC1 and remote/imported results. Missing records remain pending.
Values are percentages; mean and sample SD describe available training seeds, not independent final confirmation.
Some remote records have no local checkpoints/raw metrics; their original provenance is retained.
No setting is selected by this summary. Baseline verification and the complete tuning grid are required by the study workflow.

| Model | Variant | Seed 42 | Seed 123 | Seed 456 | n | Mean +/- sample SD |
|---|---|---:|---:|---:|---:|---:|
| DTH_SE_adLIF | augmented_v1 | pending | pending | pending | 0 | pending |
| DTH_SE_adLIF | dropout030_v1 | 90.8868 | pending | pending | 1 | 90.8868 (SD unavailable) |
| DTH_SE_adLIF | dropout035_v1 | 90.5199 | pending | 90.4587 | 2 | 90.4893 +/- 0.0432 |
| DTH_SE_adLIF | spike_delete010_v1 | 90.6422 | 91.3761 | 92.5382 | 3 | 91.5189 +/- 0.9560 |
| LG_DTH_SE_adLIF | augmented_v1 | pending | pending | pending | 0 | pending |
| LG_DTH_SE_adLIF | dropout030_v1 | 91.0092 | pending | pending | 1 | 91.0092 (SD unavailable) |
| LG_DTH_SE_adLIF | dropout035_v1 | 91.0092 | pending | 91.2538 | 2 | 91.1315 +/- 0.1730 |
| LG_DTH_SE_adLIF | spike_delete010_v1 | 90.0917 | 90.4587 | 92.3547 | 3 | 90.9684 +/- 1.2145 |
