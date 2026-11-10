# Combined SHD speaker-holdout results

Four settings, two models, three training seeds: 24 completed runs. Ordinary inference without TTA.
Values are accuracy percentages; variation is sample SD across seeds 42, 123 and 456.
Checkpoints were selected by validation accuracy. Regularization test results are exploratory.

| Setting | Model | Dropout | Spike deletion | n | Validation mean +/- SD (%) | Test mean +/- SD (%) | Selected by validation |
|---|---|---:|---:|---:|---:|---:|---|
| Augmented dropout 0.25 | DTH_SE_adLIF | 0.25 | 5% | 3 | 90.56 +/- 0.92 | 93.14 +/- 0.34 |  |
| Augmented dropout 0.25 | LG_DTH_SE_adLIF | 0.25 | 5% | 3 | 91.76 +/- 0.58 | 94.26 +/- 2.16 | Yes |
| Augmented dropout 0.30 | DTH_SE_adLIF | 0.30 | 5% | 3 | 90.46 +/- 0.74 | 92.49 +/- 1.56 |  |
| Augmented dropout 0.30 | LG_DTH_SE_adLIF | 0.30 | 5% | 3 | 91.68 +/- 0.64 | 93.48 +/- 1.52 |  |
| Augmented dropout 0.35 | DTH_SE_adLIF | 0.35 | 5% | 3 | 91.05 +/- 0.97 | 93.15 +/- 2.34 |  |
| Augmented dropout 0.35 | LG_DTH_SE_adLIF | 0.35 | 5% | 3 | 90.95 +/- 0.34 | 93.14 +/- 0.59 |  |
| Dropout 0.25 + deletion 10% | DTH_SE_adLIF | 0.25 | 10% | 3 | 91.52 +/- 0.96 | 93.29 +/- 1.43 | Yes |
| Dropout 0.25 + deletion 10% | LG_DTH_SE_adLIF | 0.25 | 10% | 3 | 90.97 +/- 1.21 | 92.98 +/- 1.25 |  |

## Per-seed results

| Setting | Model | Seed | Selected epoch (zero-based) | Train at selected epoch (%) | Final train (%) | Best validation (%) | Test (%) | Record |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Augmented dropout 0.25 | DTH_SE_adLIF | 42 | 290 | 99.55 | 99.39 | 90.03 | 92.84 | [JSON](augmented_v1/DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Augmented dropout 0.25 | DTH_SE_adLIF | 123 | 261 | 99.89 | 99.62 | 90.03 | 93.07 | [JSON](augmented_v1/DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Augmented dropout 0.25 | DTH_SE_adLIF | 456 | 140 | 99.83 | 99.44 | 91.62 | 93.51 | [JSON](augmented_v1/DTH_SE_adLIF/SHD/seed_456/RESULT.json) |
| Augmented dropout 0.25 | LG_DTH_SE_adLIF | 42 | 143 | 99.70 | 99.84 | 91.19 | 93.68 | [JSON](augmented_v1/LG_DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Augmented dropout 0.25 | LG_DTH_SE_adLIF | 123 | 94 | 99.55 | 99.78 | 92.35 | 96.64 | [JSON](augmented_v1/LG_DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Augmented dropout 0.25 | LG_DTH_SE_adLIF | 456 | 249 | 99.58 | 99.80 | 91.74 | 92.45 | [JSON](augmented_v1/LG_DTH_SE_adLIF/SHD/seed_456/RESULT.json) |
| Augmented dropout 0.30 | DTH_SE_adLIF | 42 | 107 | 99.37 | 99.72 | 90.89 | 91.92 | [JSON](dropout030_v1/DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Augmented dropout 0.30 | DTH_SE_adLIF | 123 | 205 | 99.73 | 99.28 | 89.60 | 91.30 | [JSON](dropout030_v1/DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Augmented dropout 0.30 | DTH_SE_adLIF | 456 | 276 | 99.66 | 99.70 | 90.89 | 94.26 | [JSON](dropout030_v1/DTH_SE_adLIF/SHD/seed_456/RESULT.json) |
| Augmented dropout 0.30 | LG_DTH_SE_adLIF | 42 | 157 | 99.41 | 99.83 | 91.01 | 92.09 | [JSON](dropout030_v1/LG_DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Augmented dropout 0.30 | LG_DTH_SE_adLIF | 123 | 296 | 99.72 | 99.78 | 91.74 | 93.24 | [JSON](dropout030_v1/LG_DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Augmented dropout 0.30 | LG_DTH_SE_adLIF | 456 | 228 | 99.37 | 99.67 | 92.29 | 95.10 | [JSON](dropout030_v1/LG_DTH_SE_adLIF/SHD/seed_456/RESULT.json) |
| Augmented dropout 0.35 | DTH_SE_adLIF | 42 | 213 | 99.27 | 99.53 | 90.52 | 92.01 | [JSON](dropout035_v1/DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Augmented dropout 0.35 | DTH_SE_adLIF | 123 | 145 | 99.52 | 99.75 | 92.17 | 95.85 | [JSON](dropout035_v1/DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Augmented dropout 0.35 | DTH_SE_adLIF | 456 | 147 | 99.42 | 99.77 | 90.46 | 91.61 | [JSON](dropout035_v1/DTH_SE_adLIF/SHD/seed_456/RESULT.json) |
| Augmented dropout 0.35 | LG_DTH_SE_adLIF | 42 | 297 | 99.61 | 99.62 | 91.01 | 92.89 | [JSON](dropout035_v1/LG_DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Augmented dropout 0.35 | LG_DTH_SE_adLIF | 123 | 106 | 99.44 | 99.45 | 90.58 | 93.82 | [JSON](dropout035_v1/LG_DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Augmented dropout 0.35 | LG_DTH_SE_adLIF | 456 | 193 | 98.83 | 99.52 | 91.25 | 92.71 | [JSON](dropout035_v1/LG_DTH_SE_adLIF/SHD/seed_456/RESULT.json) |
| Dropout 0.25 + deletion 10% | DTH_SE_adLIF | 42 | 122 | 99.42 | 99.64 | 90.64 | 91.70 | [JSON](spike_delete010_v1/DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Dropout 0.25 + deletion 10% | DTH_SE_adLIF | 123 | 108 | 99.25 | 99.55 | 91.38 | 93.68 | [JSON](spike_delete010_v1/DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Dropout 0.25 + deletion 10% | DTH_SE_adLIF | 456 | 197 | 99.56 | 99.28 | 92.54 | 94.48 | [JSON](spike_delete010_v1/DTH_SE_adLIF/SHD/seed_456/RESULT.json) |
| Dropout 0.25 + deletion 10% | LG_DTH_SE_adLIF | 42 | 190 | 99.94 | 99.31 | 90.09 | 91.65 | [JSON](spike_delete010_v1/LG_DTH_SE_adLIF/SHD/seed_42/RESULT.json) |
| Dropout 0.25 + deletion 10% | LG_DTH_SE_adLIF | 123 | 289 | 99.67 | 99.70 | 90.46 | 94.13 | [JSON](spike_delete010_v1/LG_DTH_SE_adLIF/SHD/seed_123/RESULT.json) |
| Dropout 0.25 + deletion 10% | LG_DTH_SE_adLIF | 456 | 124 | 99.66 | 99.64 | 92.35 | 93.15 | [JSON](spike_delete010_v1/LG_DTH_SE_adLIF/SHD/seed_456/RESULT.json) |

The selection column uses the frozen three-seed validation rule, including the dropout-0.25 baseline.
Test scores do not select settings. Previously observed test results remain exploratory.
Original protocol metadata and checkpoint identities are retained in each record and evidence snapshot.
These speaker-holdout results differ from random recording SHD*; paper accuracy is not a matched baseline.
