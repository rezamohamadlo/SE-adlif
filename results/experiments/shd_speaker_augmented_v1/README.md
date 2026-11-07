# SHD* speaker holdout + augmentation v1

Prepared on 7 November 2026; **training has not started**. Protocol:
`shd_speaker_augmented_v1`, implementation 1.0.0. This fresh-training experiment
uses the existing DTH and full LG-DTH architectures with the user-approved
dropout 0.25, time scale 0.9–1.1, channel shift ±2 and 5% spike deletion.

The user's local startup test was removed from active results on 7 November
2026 before moving to another PC. Only DTH seed 42 startup configs/logs existed;
no saved checkpoint, training metrics or final-test result was present.
Automatic approval review rejected deletion as blocked by policy, so the
startup directory and inactive launcher lock were moved to the ignored local
`archive/startup_checks/shd_speaker_augmented_v1/2026-11-07/` instead. The next
launch starts fresh; the frozen split and experiment setup are retained.

The frozen [split manifest](SPLIT_MANIFEST.json) was verified against the local
official train/test HDF5 files. Training speakers: 1, 2, 6, 7, 8, 9, 10, 11
(6,521 recordings); validation speakers: 0, 3 (1,635 recordings). Split seed 42.
Fingerprint: `e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
The split identity is inherited from the archived speaker split; this is a
distinct training protocol and output root with no checkpoint initialization.

| Position | Model | Training seed | Initial state |
|---|---|---:|---|
| 1 | DTH_SE_adLIF | 42 | Not started |
| 2 | LG_DTH_SE_adLIF | 42 | Not started |
| 3 | DTH_SE_adLIF | 123 | Not started |
| 4 | LG_DTH_SE_adLIF | 123 | Not started |
| 5 | DTH_SE_adLIF | 456 | Not started |
| 6 | LG_DTH_SE_adLIF | 456 | Not started |

This table records preparation state only. During execution, consult each
seed's metrics/config/checkpoints and the model's generated `RESULTS.md` for
current completion evidence. No validation or test accuracy is available yet.

Baseline: `config/paper_baseline.yaml`, version `paper_classification_v1`, SHD
section. The split and regularization deviations are saved in every resolved
run config. Full budget: 300 epochs, batch 256, LR 0.01, no early stopping.

See the [experiment guide](../../../docs/experiments/shd-speaker-augmented-v1.md)
for the configuration, user launch command, recovery and reporting policy.
