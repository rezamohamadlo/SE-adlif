# Reference versus six proposed models - SHD*

Snapshot: 17 October 2026. Target: **7 models x 3 training seeds = 21 baseline-matched runs**. Twenty are completed and verified locally; LG-DTH seed 456 awaits a fresh baseline rerun. Completed early-stopping results are kept separately below.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Matched completed / 3 | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|---:|
| SE-adLIF reference | 92.93 | 92.89 | 93.64 | 3 / 3 | 93.15 ± 0.42 |
| DTH | 94.70 | 94.88 | 93.90 | 3 / 3 | 94.49 ± 0.52 |
| LG-DTH | 93.77 | 93.02 | pending fresh rerun | 2 / 3 | 93.40 ± 0.53 |
| DA | 93.73 | 93.99 | 92.27 | 3 / 3 | 93.33 ± 0.93 |
| FP | 93.51 | 92.49 | 94.04 | 3 / 3 | 93.35 ± 0.79 |
| MR (K=2) | 94.04 | 93.95 | 93.68 | 3 / 3 | 93.89 ± 0.18 |
| MT | 94.88 | 94.52 | 94.43 | 3 / 3 | 94.61 ± 0.23 |

Means and sample SDs use only the completed matched seeds shown in each row. LG-DTH currently has n=2; the other rows have n=3. No repeated seed is counted as an independent training seed.

| Model | Mean difference from reference over matching seeds (pp) |
|---|---:|
| DTH | +1.34 |
| LG-DTH (seeds 42/123 only) | +0.49 |
| DA | +0.18 |
| FP | +0.19 |
| MR (K=2) | +0.74 |
| MT | +1.46 |

Differences use unrounded metrics and matching training seeds. MT has the highest observed three-seed mean. Three seeds do not establish statistical significance.

| Run / status | Details |
|---|---|
| DTH | All three completed runs were copied from the other computer. Configs pass runner guards; last checkpoints reach epoch 299 with optimizer state, and validation-selected best checkpoints are present. |
| DA, FP, MT, MR | All 12 replacement runs completed with batch 256 and early stopping disabled. Older batch-512 results and historical mixed histories are excluded. |
| LG-DTH seeds 42/123 | Completed matched baseline runs; retained unchanged. |
| LG-DTH seed 456 fresh rerun | Interrupted and ready to resume at the canonical `results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/seed_456/` directory. Last checkpoint: epoch 115, step 2900; continue toward 300 epochs with early stopping disabled. Original weights/optimizer/scheduler state preserved. |
| Rerun provenance | Protocol-corrected fresh rerun after viewing the earlier test result. Retain that disclosure; use validation alone for checkpoint selection, then evaluate the official test. Do not choose between repeated seed-456 runs using their test scores. |

| Separate historical result | Test accuracy (%) | Early stopping | Included in matched table |
|---|---:|---|---|
| LG-DTH seed 456 | 93.51 | Enabled; 105 completed epochs | No |

The earlier early-stopping run is preserved at `results/archive/lg_dth_seed456_early_stopping_20261017/`. Its mixed-stopping three-seed aggregate with seeds 42/123 is **93.43 ± 0.38%**, descriptive only. The fresh rerun now occupies the canonical seed directory; its final test result remains pending.

| Shared setting | Paper-baseline requirement |
|---|---|
| Authority | [config/paper_baseline.yaml](../config/paper_baseline.yaml), SHD section, `paper_classification_v1` |
| Seeds | Training: 42, 123, 456; fixed split seed 42 |
| Selection | 20% official training data for validation; best validation-accuracy checkpoint; official test afterward |
| Optimization | Adam; LR 0.01; batch 256; gradient clipping 1.5 |
| Budget / scheduler | 300 total epochs; early stopping disabled; plateau patience 9999, factor 0.9 |
| Shared architecture | Two 360-neuron hidden layers; dropout 0.15; intended neuron mechanisms differ |
| Preprocessing | Spatial factor 0.2; time factor 0.001; window 4; minimum padding 250; ignore first 10 timesteps |

| Evidence | Source |
|---|---|
| Reference | [Canonical records](../results/valid/reference/SHD/REFERENCE_RESULTS.json) |
| DTH | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/DTH_SE_adLIF/SHD/) |
| LG-DTH | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/) |
| DA | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/DA_SE_adLIF/SHD/) |
| FP | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/FP_SE_adLIF/SHD/) |
| MR (K=2) | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/MR_SE_adLIF/K_2/SHD/) |
| MT | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/MT_SE_adLIF/SHD/) |

[LG-DTH detailed results](../results/valid/protocol_v2/LG_DTH_SE_adLIF/RESULTS.md) | [Paper versus reference](paper-vs-reference.md)
