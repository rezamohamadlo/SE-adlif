# ECG transfer - 19 October 2026

Four run directories were moved from user-specified external storage `F:/reza/SE-adlif` into their canonical repository-relative destinations. Because this crosses volumes, each directory was transferred, every destination file hash was compared against the preflight inventory, the source inventory was rechecked, and only then was its source directory removed. All 56 original files passed verification.

The four destination seed directories did not previously exist. No active local results were deleted or overwritten. The existing older early-stopped runs remain under `archive/unmatched_early_stopping_20261017/`; no additional archive move was necessary. The historical SE-adLIF ECG reference was preserved.

| Destination | Status |
|---|---|
| [results/valid/DTH_SE_adLIF/ECG/seed_42](../../results/valid/DTH_SE_adLIF/ECG/seed_42/TRAINING_SUMMARY.md) | moved; source removed; original hashes verified |
| [results/valid/DTH_SE_adLIF/ECG/seed_123](../../results/valid/DTH_SE_adLIF/ECG/seed_123/TRAINING_SUMMARY.md) | moved; source removed; original hashes verified |
| [results/valid/MT_SE_adLIF/ECG/seed_123](../../results/valid/MT_SE_adLIF/ECG/seed_123/TRAINING_SUMMARY.md) | moved; source removed; original hashes verified |
| [results/valid/MT_SE_adLIF/ECG/seed_456](../../results/valid/MT_SE_adLIF/ECG/seed_456/TRAINING_SUMMARY.md) | moved; source removed; original hashes verified |

Per-seed summaries/configs/metrics/checkpoints and cost evidence were retained byte-for-byte. Model accuracy/cost tables, the canonical-results README and general comparison report were updated. No training or cost evaluation was launched.

The four SSC transfer directories were outside this ECG replacement action and remain on the external drive. DTH SSC checkpoints are therefore still missing from this checkout; MT SSC missing completions have not yet been imported.

[File-by-file move manifest](../../results/provenance/ecg-transfer-2026-10-19.json). Historical source paths are retained solely as provenance; destination artifact paths are repository-relative.

Subsequent SSC move completed: all four remaining SSC folders have now been moved into this checkout, with preexisting local evidence archived and omitted DTH metrics retained. No transferred seed folders remain at the external source. See the [SSC transfer record](ssc-transfer-2026-10-19.md). Earlier descriptions of remaining external SSC folders are historical snapshots.
