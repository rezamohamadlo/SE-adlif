# Reference / DTH / MT comparison status - 20 October 2026

21/27 accepted completed results: 18 matched and 3 accepted historical ECG reference runs with unmatched stopping/budget. Six SSC runs are missing locally.

| Model | SHD* accepted | SSC mean-loss accepted | ECG accepted | Total | Matched |
|---|---|---|---|---:|---:|
| Reference | 42, 123, 456 | 42 | 42, 123, 456* | 7/9 | 4/9 |
| DTH | 42, 123, 456 | 42 | 42, 123, 456 | 7/9 | 7/9 |
| MT | 42, 123, 456 | 42 | 42, 123, 456 | 7/9 | 7/9 |

*Reference ECG is accepted historical evidence; cross-model ECG comparisons remain descriptive.

| Model | Dataset | Seed | Test (%) | Status | Evidence |
|---|---|---:|---:|---|---|
| reference | SHD* | 42 | 92.93 | completed_matched | [42](reference/SHD/seed_42/TRAINING_SUMMARY.md) |
| reference | SHD* | 123 | 92.89 | completed_matched | [123](reference/SHD/seed_123/TRAINING_SUMMARY.md) |
| reference | SHD* | 456 | 93.64 | completed_matched | [456](reference/SHD/seed_456/TRAINING_SUMMARY.md) |
| reference | SSC | 42 | 80.69 | completed_matched | [42](reference/SSC/seed_42/TRAINING_SUMMARY.md) |
| reference | SSC | 123 | pending | missing | not available locally |
| reference | SSC | 456 | pending | missing | not available locally |
| reference | ECG | 42 | 88.13 | accepted_historical_unmatched | [42](reference/ECG/seed_42/TRAINING_SUMMARY.md) |
| reference | ECG | 123 | 88.72 | accepted_historical_unmatched | [123](reference/ECG/seed_123/TRAINING_SUMMARY.md) |
| reference | ECG | 456 | 88.11 | accepted_historical_unmatched | [456](reference/ECG/seed_456/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SHD* | 42 | 94.70 | completed_matched | [42](DTH_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SHD* | 123 | 94.88 | completed_matched | [123](DTH_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SHD* | 456 | 93.90 | completed_matched | [456](DTH_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SSC | 42 | 80.46 | completed_matched | [42](DTH_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SSC | 123 | pending | missing | not available locally |
| DTH_SE_adLIF | SSC | 456 | pending | missing | not available locally |
| DTH_SE_adLIF | ECG | 42 | 87.81 | completed_matched | [42](DTH_SE_adLIF/ECG/seed_42/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | ECG | 123 | 87.81 | completed_matched | [123](DTH_SE_adLIF/ECG/seed_123/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | ECG | 456 | 87.27 | completed_matched | [456](DTH_SE_adLIF/ECG/seed_456/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SHD* | 42 | 94.88 | completed_matched | [42](MT_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SHD* | 123 | 94.52 | completed_matched | [123](MT_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SHD* | 456 | 94.43 | completed_matched | [456](MT_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SSC | 42 | 80.01 | completed_matched | [42](MT_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SSC | 123 | pending | missing | not available locally |
| MT_SE_adLIF | SSC | 456 | pending | missing | not available locally |
| MT_SE_adLIF | ECG | 42 | 87.26 | completed_matched | [42](MT_SE_adLIF/ECG/seed_42/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | ECG | 123 | 88.19 | completed_matched | [123](MT_SE_adLIF/ECG/seed_123/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | ECG | 456 | 87.98 | completed_matched | [456](MT_SE_adLIF/ECG/seed_456/TRAINING_SUMMARY.md) |

Remaining: SSC seeds 123 and 456 for each model. Other-computer runs count only after verification/import. [Machine-readable status](COMPARISON_STATUS.json).
