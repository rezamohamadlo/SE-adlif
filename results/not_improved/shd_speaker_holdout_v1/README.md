# Archived SHD speaker-holdout experiment

Retired at the user request on 5 November 2026. Protocol: `shd_speaker_holdout_v1`.
Original root: `results/experiments/shd_speaker_holdout_v1/`.
Archived root: `results/not_improved/shd_speaker_holdout_v1/`.

| Model | Seed | Status | Last logged epoch (zero-based) | Final test accuracy (%) |
|---|---:|---|---:|---:|
| DTH_SE_adLIF | 42 | complete | 300.0 | 92.58 |
| DTH_SE_adLIF | 123 | incomplete | 237.0 | Pending |
| DTH_SE_adLIF | 456 | not_started | None | Pending |
| LG_DTH_SE_adLIF | 42 | complete | 300.0 | 92.76 |
| LG_DTH_SE_adLIF | 123 | not_started | None | Pending |
| LG_DTH_SE_adLIF | 456 | not_started | None | Pending |

Epoch 300 in completed CSV histories is a final-test logging step; training used epochs 0 through 299. The partial run has no final test.

This is a discontinued exploratory comparison, not a completed three-seed result or proof that either architecture fails. Seed-42 speaker-holdout test accuracy was lower than the earlier random-recording SHD* result for both models. These protocols select checkpoints using different validation subsets.

All original metrics, resolved configs, split manifests, summaries, cost records and best/last checkpoints are preserved without rewriting their contents. `ARCHIVE.json` inventories every original file by path, size and SHA-256; the two new archive documentation files are excluded. Historical artifact prefixes resolve through `config/path_aliases.yaml`.

The phase launcher now refuses to restart this retired workflow. Intentional reruns require a new experiment version and output directory.

[Experiment guide](../../../docs/experiments/shd-speaker-holdout.md)
