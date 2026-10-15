# Reference versus six proposed models - SHD*

Snapshot: 15 October 2026. Current scope: **7 models x 3 training seeds = 21 runs**. Completed final-test results are shown below; mixed stopping rules are explicitly marked. Pending and historical batch-512 runs remain excluded.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Completed / 3 | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|---:|
| SE-adLIF reference | 92.93 | 92.89 | 93.64 | 3 / 3 | 93.15 ± 0.42 |
| DTH | pending | pending | pending | 0 / 3 | pending |
| LG-DTH | 93.77 | 93.02 | 93.51 [ES] | 3 / 3 | 93.43 ± 0.38 [ES] |
| DA | pending | pending | pending | 0 / 3 | pending |
| FP | pending | pending | pending | 0 / 3 | pending |
| MR (K=2) | pending | pending | pending | 0 / 3 | pending |
| MT | pending | pending | pending | 0 / 3 | pending |

[ES] Seed 456 used early stopping (105 completed epochs); seeds 42 and 123 completed 300 epochs without early stopping. The three-seed LG-DTH aggregate is descriptive, not a fully matched paper-baseline comparison. The matched two-seed subset remains **93.40 ± 0.53%**.

| Model / evidence | Status and interpretation |
|---|---|
| [Reference](../results/valid/reference/SHD/REFERENCE_RESULTS.json) | Three completed canonical SHD* runs; do not substitute old dated reference scores. |
| DTH | User reports baseline training on another computer. Handled externally; completion and final test results remain pending local verification/import. Local seed-42 files conflict, so no local DTH score is admitted. |
| [LG-DTH](../results/valid/protocol_v2/LG_DTH_SE_adLIF/RESULTS.md) | Seeds 42 and 123 completed with matched baseline settings. Seed 456 completed with early stopping enabled (93.51% test); it is included only in the descriptive three-seed aggregate, not the matched baseline subset. |
| DA, FP, MR, MT | Previous SHD* seed-42 runs used batch 512 and early stopping; their scores are excluded. Three fresh baseline-matched seeds are required per model (12 runs). |
| [LG-DTH seed 456, early stopping](../results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/seed_456/FULL_RUN_CONFIG.json) | Completed early-stopping experiment: 93.51% final test accuracy. Relocated into the main model directory with its original protocol retained. The earlier interrupted baseline attempt is archived separately. Its stopping-rule deviation is marked [ES] in the main table. |
| History | MT SHD* has overlapping-history warnings. Historical scores remain archived evidence and are not independent matched comparison results. |

| Shared setting | Paper-baseline value (seed 456 stopping deviation noted above) |
|---|---|
| Configuration authority | [config/paper_baseline.yaml](../config/paper_baseline.yaml), SHD section, version `paper_classification_v1` |
| Training seeds / split seed | Training: 42, 123, 456; fixed split: 42 |
| Split and selection | 20% official training data for validation; best validation-accuracy checkpoint; official test afterward |
| Optimization | Adam; LR 0.01; batch 256; gradient clipping 1.5 |
| Budget / scheduler | 300 total epochs; early stopping disabled; plateau patience 9999, factor 0.9 |
| Shared architecture | Two 360-neuron hidden layers; dropout 0.15; intended neuron mechanisms differ |
| Preprocessing | Spatial factor 0.2; time factor 0.001; window 4; minimum padding 250; ignore first 10 timesteps |

| Per-seed comparison | Proposed test (%) | Reference test (%) | Difference (pp) |
|---|---:|---:|---:|
| LG-DTH, seed 42 | 93.77 | 92.93 | +0.84 |
| LG-DTH, seed 123 | 93.02 | 92.89 | +0.13 |
| LG-DTH, seed 456 [ES] | 93.51 | 93.64 | -0.13 |

Differences use unrounded test metrics and matching training seeds. LG-DTH now has three completed test evaluations. The seed-456 difference is descriptive because its stopping rule differs from the reference. Six completed evaluations are verified locally, of which five match the baseline stopping rule (three reference and LG-DTH seeds 42/123). No missing test score is inferred from validation accuracy.

[LG-DTH seed-456 metrics](../results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) | [LG-DTH seed-123 metrics](../results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv) | [LG-DTH seed-42 metrics](../results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv) · [Archived audit](archive/2026-10-15-report-consolidation/training-baseline-audit.md) · [Paper versus reference](paper-vs-reference.md)
