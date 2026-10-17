# Reference versus six proposed models - SHD*

Snapshot: 17 October 2026. Target: **7 models x 3 training seeds = 21 baseline-matched runs**. Twenty are completed and verified locally; the fresh LG-DTH seed-456 rerun is in progress; its final test result is pending. Completed early-stopping results are kept separately below.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Matched completed / 3 | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|---:|
| SE-adLIF reference | 92.93 | 92.89 | 93.64 | 3 / 3 | 93.15 ± 0.42 |
| DTH | 94.70 | 94.88 | 93.90 | 3 / 3 | 94.49 ± 0.52 |
| LG-DTH | 93.77 | 93.02 | pending (training) | 2 / 3 | 93.40 ± 0.53 |
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
| LG-DTH seed 456 fresh rerun | Training in progress at the canonical directory. 252/300 epochs completed (latest validation epoch 251, 98.96%); best validation 99.02% at epoch 226. Final test pending. Snapshot: 2026-10-17 10:56 Tehran. Early stopping disabled; resumed optimizer/scheduler state retained. |
| Rerun provenance | Protocol-corrected fresh rerun after viewing the earlier test result. Retain that disclosure; use validation alone for checkpoint selection, then evaluate the official test. Do not choose between repeated seed-456 runs using their test scores. |

| Separate historical result | Test accuracy (%) | Early stopping | Included in matched table |
|---|---:|---|---|
| LG-DTH seed 456 | 93.51 | Enabled; 105 completed epochs | No |

The earlier early-stopping run is preserved at `results/archive/lg_dth_seed456_early_stopping_20261017/`. Its mixed-stopping three-seed aggregate with seeds 42/123 is **93.43 ± 0.38%**, descriptive only. The fresh rerun now occupies the canonical seed directory; its final test result remains pending.

## Spike activity and inference cost

The parameter counts and dense matrix-compute estimates below are recorded in each model's per-seed `TRAINING_SUMMARY.md`; they agree across seeds 42, 123 and 456. LG-DTH seed 456 is still training, so its entries describe architecture and recorded validation computation, not a completed final-test evaluation.

| Model | Total parameters | Trainable parameters | Estimated dense GFLOPs / validation sample |
|---|---:|---:|---:|
| SE-adLIF reference | 450,760 | 450,020 | 0.2619 |
| DTH | 451,484 | 450,744 | 0.2619 |
| LG-DTH | 452,924 | 452,184 | 0.2619 |
| DA | 453,640 | 452,900 | 0.2619 |
| FP | 450,760 | 450,020 | 0.2619 |
| MR (K=2) | 450,760 | 450,020 | 0.2619 |
| MT | 452,200 | 451,460 | 0.2619 |

The existing estimate uses 293.31 average padded validation timesteps/sample and counts feed-forward and recurrent matrix multiply-adds as two FLOPs each. It excludes elementwise neuron updates, including the additional operations of proposed variants, and is not a complete compute or energy estimate. The shared matrix dimensions explain the identical estimates.

| Model | Spikes/sample, L1 / L2 | Firing fraction (%), L1 / L2 | Estimated synaptic operations/sample | Inference time (ms/sample) |
|---|---|---|---|---|
| SE-adLIF reference | Pending | Pending | Pending | Pending |
| DTH | Pending | Pending | Pending | Pending |
| LG-DTH | Pending | Pending | Pending | Pending |
| DA | Pending | Pending | Pending | Pending |
| FP | Pending | Pending | Pending | Pending |
| MR (K=2) | Pending | Pending | Pending | Pending |
| MT | Pending | Pending | Pending | Pending |

Pending means not measured: existing accuracy logs and checkpoints do not contain these statistics. Do not interpret pending as zero or infer firing activity from parameter counts.

For the additional measurements, evaluate each seed's best validation-selected checkpoint in inference mode on the same SHD* evaluation split, with matching batching and padding. Report per-seed values and mean +/- sample SD over matching completed seeds. Count emitted spikes before dropout, separately for each hidden layer; firing fraction is spike count divided by valid neuron-timestep opportunities. Report valid-sequence activity separately from padded-timestep activity. Estimate synaptic operations from actual presynaptic activity and connection fan-out, including recurrent connections and input events; exclude final-step spikes that have no subsequent recurrent use. Describe the connectivity and event-counting assumptions explicitly.

Measure runtime separately from instrumented spike counting, after warm-up, on the same device, software environment and batch size, with GPU synchronization. State whether data transfer is included. Spike-derived synaptic operations describe an event-driven execution estimate; the current implementation performs dense matrix operations each timestep. Neither statistic establishes energy savings without an energy measurement.

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
