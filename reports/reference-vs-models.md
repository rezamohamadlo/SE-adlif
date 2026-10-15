# Reference versus six proposed models - SHD*

Snapshot: 15 October 2026. Current scope: **7 models x 3 training seeds = 21 runs**. Only completed final-test results with matched saved settings enter the accuracy table. Pending and historical unmatched runs do not count as completed matched runs.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Completed / 3 | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|---:|
| SE-adLIF reference | 92.93 | 92.89 | 93.64 | 3 / 3 | 93.15 ± 0.42 |
| DTH | pending | pending | pending | 0 / 3 | pending |
| LG-DTH | 93.77 | pending | pending | 1 / 3 | pending |
| DA | pending | pending | pending | 0 / 3 | pending |
| FP | pending | pending | pending | 0 / 3 | pending |
| MR (K=2) | pending | pending | pending | 0 / 3 | pending |
| MT | pending | pending | pending | 0 / 3 | pending |

| Model / evidence | Status and interpretation |
|---|---|
| [Reference](../results/valid/reference/SHD/REFERENCE_RESULTS.json) | Three completed canonical SHD* runs; do not substitute old dated reference scores. |
| DTH | User reports baseline training on another computer. Handled externally; completion and final test results remain pending local verification/import. Local seed-42 files conflict, so no local DTH score is admitted. |
| [LG-DTH](../results/valid/protocol_v2/LG_DTH_SE_adLIF/RESULTS.md) | 1 completed matched run(s) locally. Seeds without a completed final test remain pending; live metrics can be newer than summaries. |
| DA, FP, MR, MT | Previous SHD* seed-42 runs used batch 512 and early stopping; their scores are excluded. Three fresh baseline-matched seeds are required per model (12 runs). |
| History | MT SHD* has overlapping-history warnings. Historical scores remain archived evidence and are not independent matched comparison results. |

| Shared setting | Required value for every model |
|---|---|
| Configuration authority | [config/paper_baseline.yaml](../config/paper_baseline.yaml), SHD section, version `paper_classification_v1` |
| Training seeds / split seed | Training: 42, 123, 456; fixed split: 42 |
| Split and selection | 20% official training data for validation; best validation-accuracy checkpoint; official test afterward |
| Optimization | Adam; LR 0.01; batch 256; gradient clipping 1.5 |
| Budget / scheduler | 300 total epochs; early stopping disabled; plateau patience 9999, factor 0.9 |
| Shared architecture | Two 360-neuron hidden layers; dropout 0.15; intended neuron mechanisms differ |
| Preprocessing | Spatial factor 0.2; time factor 0.001; window 4; minimum padding 250; ignore first 10 timesteps |

| Matched seed comparison | Proposed test (%) | Reference test (%) | Difference (pp) |
|---|---:|---:|---:|
| LG-DTH, seed 42 | 93.77 | 92.93 | +0.84 |

Differences use unrounded test metrics and matching training seeds. A single completed proposed-model seed does not establish a three-seed improvement. No missing test score is inferred from validation accuracy.

[LG-DTH seed-42 metrics](../results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv) · [Archived audit](archive/2026-10-15-report-consolidation/training-baseline-audit.md) · [Paper versus reference](paper-vs-reference.md)
