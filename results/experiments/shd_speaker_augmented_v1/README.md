# SHD speaker holdout + augmentation v1

## Completed results (8 November 2026)

All six fresh-training runs completed 300 epochs, final official-test
evaluation and per-seed cost measurement. Checkpoints were selected by held-out
training-speaker validation accuracy. Ordinary inference uses no TTA.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | n | Mean +/- sample SD (%) |
|---|---:|---:|---:|---:|---:|
| DTH | 92.8445 | 93.0654 | 93.5071 | 3 | 93.1390 +/- 0.3374 |
| LG-DTH | 93.6837 | 96.6431 | 92.4470 | 3 | 94.2580 +/- 2.1562 |

The underlying `metrics.csv`, resolved configs, best/last checkpoints and
measurement evidence remain beside each seed. Raw metrics, logs, completion
markers and weights are transferred separately because Git ignores them.
Model-level `RESULTS.md` files contain the ordinary aggregates.

The separate [exploratory TTA evaluation](../shd_speaker_augmented_tta_v1/RESULTS.md)
uses these same validation-selected checkpoints, fixed input gains
1.00/0.95/1.05 and mean class probabilities. It performs no training. TTA test
means are 93.1095 +/- 0.3851% for DTH and 94.4641 +/- 2.0724% for LG-DTH
(sample SD, n=3). Those results do not replace ordinary inference evidence.

This speaker-holdout protocol differs from the random recording split used
for the paper SHD* comparison. One fixed speaker pair and three training seeds
do not establish robustness across arbitrary speaker splits.

## Historical preparation record (7 November 2026)

The following records the state before the completed runs above.

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
