# MR_SE_adLIF: full evaluation

Adaptation update interval K=2.
Statistics use final test accuracy from completed runs only.
Target comparison protocol: paper_classification_v1; historical exceptions are identified below. Baseline: config/paper_baseline.yaml; dataset settings: config/comparison/.
SHD* training-derived validation: 20%; ECG: 5%; split seeds: SHD* 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/256, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* | 94.04% | 93.95% | 93.68% | 3 | 93.89% ± 0.18% |
| SSC | 76.93% | pending | pending | 1 | pending |
| ECG (historical, unmatched) | 88.05% | pending | pending | 1 | pending |

## Evidence audit - 18 October 2026

The ECG seed-42 score is historical: its saved config enables early stopping, unlike the current baseline. It is excluded from matched ECG comparisons; no matched ECG completion is established by this table.
