# Canonical SE-adLIF reference - SHD*

Baseline: paper_classification_v1. Batch 256, LR 0.01, dropout 0.15, 300 epochs, early stopping disabled.
Validation: 20% of training, split seed 42. Final test uses the best validation checkpoint.

| Seed | Best epoch | Validation accuracy | Test accuracy | Test loss |
|---|---:|---:|---:|---:|
| 42 | 261 | 99.14% | 92.93% | 0.6220 |
| 123 | 204 | 99.39% | 92.89% | 0.6165 |
| 456 | 243 | 99.08% | 93.64% | 0.5895 |

Mean ± sample SD: **93.15% ± 0.42%** (3 seeds).
Paper: 93.79% ± 0.76% (20 runs); difference of means: -0.64 percentage points.

Canonical records are in seed_<seed>/. Prior controlled-v2 SHD results are retired.
The prior numerical context remains only in historical experiment REFERENCE_BASELINE.json files.
New proposed-model comparisons must match this baseline; old controlled-v2 runs are unmatched.
