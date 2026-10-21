# Reference / DTH / MT comparison status - 21 October 2026

27/27 accepted completed results: 24 matched and 3 accepted historical ECG reference runs with unmatched stopping/budget. All expected combinations are available locally.

| Model | SHD* accepted | SSC mean-loss accepted | ECG accepted | Total | Matched |
|---|---|---|---|---:|---:|
| Reference | 42, 123, 456 | 42, 123, 456 | 42, 123, 456* | 9/9 | 6/9 |
| DTH | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 9/9 | 9/9 |
| MT | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 9/9 | 9/9 |

*Reference ECG is accepted historical evidence; cross-model ECG comparisons remain descriptive.

| Model | Dataset | Seed | Test (%) | Status | Evidence |
|---|---|---:|---:|---|---|
| reference | SHD* | 42 | 92.93 | completed_matched | [42](reference/SHD/seed_42/TRAINING_SUMMARY.md) |
| reference | SHD* | 123 | 92.89 | completed_matched | [123](reference/SHD/seed_123/TRAINING_SUMMARY.md) |
| reference | SHD* | 456 | 93.64 | completed_matched | [456](reference/SHD/seed_456/TRAINING_SUMMARY.md) |
| reference | SSC | 42 | 80.69 | completed_matched | [42](reference/SSC/seed_42/TRAINING_SUMMARY.md) |
| reference | SSC | 123 | 80.33 | completed_matched | [123](reference/SSC/seed_123/REFERENCE_RESULT.json) |
| reference | SSC | 456 | 80.38 | completed_matched | [456](reference/SSC/seed_456/REFERENCE_RESULT.json) |
| reference | ECG | 42 | 88.13 | accepted_historical_unmatched | [42](reference/ECG/seed_42/TRAINING_SUMMARY.md) |
| reference | ECG | 123 | 88.72 | accepted_historical_unmatched | [123](reference/ECG/seed_123/TRAINING_SUMMARY.md) |
| reference | ECG | 456 | 88.11 | accepted_historical_unmatched | [456](reference/ECG/seed_456/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SHD* | 42 | 94.70 | completed_matched | [42](DTH_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SHD* | 123 | 94.88 | completed_matched | [123](DTH_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SHD* | 456 | 93.90 | completed_matched | [456](DTH_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SSC | 42 | 80.46 | completed_matched | [42](DTH_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | SSC | 123 | 79.84 | completed_matched | [123](DTH_SE_adLIF/SSC/seed_123/RESULT.json) |
| DTH_SE_adLIF | SSC | 456 | 79.56 | completed_matched | [456](DTH_SE_adLIF/SSC/seed_456/RESULT.json) |
| DTH_SE_adLIF | ECG | 42 | 87.81 | completed_matched | [42](DTH_SE_adLIF/ECG/seed_42/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | ECG | 123 | 87.81 | completed_matched | [123](DTH_SE_adLIF/ECG/seed_123/TRAINING_SUMMARY.md) |
| DTH_SE_adLIF | ECG | 456 | 87.27 | completed_matched | [456](DTH_SE_adLIF/ECG/seed_456/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SHD* | 42 | 94.88 | completed_matched | [42](MT_SE_adLIF/SHD/seed_42/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SHD* | 123 | 94.52 | completed_matched | [123](MT_SE_adLIF/SHD/seed_123/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SHD* | 456 | 94.43 | completed_matched | [456](MT_SE_adLIF/SHD/seed_456/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SSC | 42 | 80.01 | completed_matched | [42](MT_SE_adLIF/SSC/seed_42/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | SSC | 123 | 79.98 | completed_matched | [123](MT_SE_adLIF/SSC/seed_123/RESULT.json) |
| MT_SE_adLIF | SSC | 456 | 80.23 | completed_matched | [456](MT_SE_adLIF/SSC/seed_456/RESULT.json) |
| MT_SE_adLIF | ECG | 42 | 87.26 | completed_matched | [42](MT_SE_adLIF/ECG/seed_42/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | ECG | 123 | 88.19 | completed_matched | [123](MT_SE_adLIF/ECG/seed_123/TRAINING_SUMMARY.md) |
| MT_SE_adLIF | ECG | 456 | 87.98 | completed_matched | [456](MT_SE_adLIF/ECG/seed_456/TRAINING_SUMMARY.md) |

No missing runs. Imported SSC seeds 123/456 passed SHA-256 verification; see [import provenance](../provenance/ssc-local-import-2026-10-21.json). [Machine-readable status](COMPARISON_STATUS.json).
