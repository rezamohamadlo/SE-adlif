# Canonical SE-adLIF reference - SHD*

Final test results from the three canonical two-layer SE-adLIF reference runs. Checkpoints were selected by validation accuracy before evaluation on the official test set. Best epochs below are zero-based.

| Dataset | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Completed | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|---:|
| SHD* | 92.93 | 92.89 | 93.64 | 3 | 93.15 ± 0.42 |

| Training seed | Best validation epoch | Validation accuracy (%) | Final test accuracy (%) | Test loss | Verified record |
|---|---:|---:|---:|---:|---|
| 42 | 261 | 99.14 | 92.93 | 0.622011 | [Seed 42](seed_42/REFERENCE_RESULT.json) |
| 123 | 204 | 99.39 | 92.89 | 0.616454 | [Seed 123](seed_123/REFERENCE_RESULT.json) |
| 456 | 243 | 99.08 | 93.64 | 0.589467 | [Seed 456](seed_456/REFERENCE_RESULT.json) |

| Paper comparison | Mean ± SD (%) | Local minus paper mean (pp) |
|---|---:|---:|
| Local SHD* reference | 93.15 ± 0.42 (sample SD) | -0.64 |
| Paper, two-layer SE-adLIF SHD* | 93.79 ± 0.76 (reported SD) | reference value |

Paper source: [Table 2, PDF page 11](../../../../reference_paper.pdf#page=11). Local statistics use three distinct training seeds; the paper reports 20 runs. Differences use unrounded local accuracies and the published rounded paper mean. The paper's test-selected SHD score is not the SHD* comparison target.

| Shared configuration | Saved setting |
|---|---|
| Baseline identity | `paper_classification_v1`; [paper baseline](../../../../config/paper_baseline.yaml), SHD section |
| Recorded protocol | `paper_shd_reproduction`; original provenance retained |
| Architecture | Two recurrent hidden layers, 360 neurons per layer; SE-adLIF cells |
| Training | Batch 256; Adam, LR 0.01; 300 epochs; early stopping disabled |
| Dropout / clipping | 0.15 / 1.5 |
| Scheduler | Plateau; patience 9999; factor 0.9 |
| Validation | 20% of official training data; fixed split seed 42 |
| Training seeds | 42, 123, 456 |
| Preprocessing | Spatial factor 0.2; time factor 0.001; window 4; minimum padding 250; ignore first 10 timesteps |

[Aggregate provenance](REFERENCE_RESULTS.json) and per-seed `REFERENCE_RESULT.json` / `TEST_RESULTS.json` are the authoritative records; checkpoint paths and SHA-256 hashes are retained there. Scores above were cross-checked between these records. Canonical records are under `seed_<seed>/`; the old dated controlled-v2 SHD records are historical and excluded. Historical experiment `REFERENCE_BASELINE.json` files preserve their original comparison context.

Shared settings match the current paper baseline; this does not establish exact paper reproduction or guarantee an individual seed's accuracy. Proposed-model comparisons must match the shared settings and training seeds, with deviations reported explicitly.
