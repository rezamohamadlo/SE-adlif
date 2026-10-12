# MR_SE_adLIF: full evaluation

Adaptation update interval K=2.
Statistics use final test accuracy from completed runs only.
Controlled v2: SHD uses 20% training-derived validation; ECG uses 5%; split seed is 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Protocol: SHD 300/512, SSC 40/256, ECG 400/64 (epochs/batch).
LR scheduler patience 9999; SHD/ECG early stopping patience 50, min_delta 0.001; SSC fixed 40 epochs.

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD | 92.58% | pending | pending | 1 | pending |
| SSC | 76.93% | pending | pending | 1 | pending |
| ECG | 88.05% | pending | pending | 1 | pending |
