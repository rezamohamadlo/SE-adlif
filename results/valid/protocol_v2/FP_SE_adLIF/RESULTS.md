# FP_SE_adLIF: full evaluation

Statistics use final test accuracy from completed runs only.
Protocol: controlled_v2. Settings: evaluation.yaml.
SHD training-derived validation: 20%; ECG: 5%; split seeds: SHD 42, ECG 42.
SSC uses official train/valid/test. Test evaluation follows validation checkpoint selection.

Epochs/batch: SHD 300/512, SSC 40/256, ECG 400/64
Training settings: SHD: LR=0.01, early stopping=True, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; SSC: LR=0.006, early stopping=False, patience=50, delta=0.001, scheduler patience=9999, factor=0.9; ECG: LR=0.01, early stopping=True, patience=50, delta=0.001, scheduler patience=9999, factor=0.9

| Dataset | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD | 91.56% | pending | pending | 1 | pending |
| SSC | 77.67% | pending | pending | 1 | pending |
| ECG | 86.85% | pending | pending | 1 | pending |
