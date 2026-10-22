# Reference versus proposed models

SHD* records for DA, FP, LG-DTH, MR and MT-DTH are retained in [not_improved](../results/not_improved/README.md); their valid measurements remain included here. Whole model folders are retained there; the classification is based on SHD* accuracy, not separate SSC/ECG outcome assessments.

Snapshot: 20 October 2026. SHD* accuracy values were rechecked; existing cost measurements were not rerun. **8 models x 3 training seeds = 24 completed SHD* runs**, including the MT-DTH combination follow-up. The original seven-model suite is retained; MT-DTH was proposed after observing its results and is a disclosed follow-up, not a preregistered independent architecture comparison. Completed early-stopping results are kept separately below.

| Model | Seed 42 test (%, Δ paper) | Seed 123 test (%, Δ paper) | Seed 456 test (%, Δ paper) | Matched completed / 3 | Mean Â± sample SD (%, Δ paper) |
|---|---:|---:|---:|---:|---:|
| SE-adLIF reference | 92.93 (-0.86 pp) | 92.89 (-0.90 pp) | 93.64 (-0.15 pp) | 3 / 3 | 93.15 Â± 0.42 (-0.64 pp) |
| DTH | 94.70 (+0.91 pp) | 94.88 (+1.09 pp) | 93.90 (+0.11 pp) | 3 / 3 | 94.49 Â± 0.52 (+0.70 pp) |
| LG-DTH | 93.77 (-0.02 pp) | 93.02 (-0.77 pp) | 94.43 (+0.64 pp) | 3 / 3 | 93.74 Â± 0.71 (-0.05 pp) |
| DA | 93.73 (-0.06 pp) | 93.99 (+0.20 pp) | 92.27 (-1.52 pp) | 3 / 3 | 93.33 Â± 0.93 (-0.46 pp) |
| FP | 93.51 (-0.28 pp) | 92.49 (-1.30 pp) | 94.04 (+0.25 pp) | 3 / 3 | 93.35 Â± 0.79 (-0.44 pp) |
| MR (K=2) | 94.04 (+0.25 pp) | 93.95 (+0.16 pp) | 93.68 (-0.11 pp) | 3 / 3 | 93.89 Â± 0.18 (+0.10 pp) |
| MT | 94.88 (+1.09 pp) | 94.52 (+0.73 pp) | 94.43 (+0.64 pp) | 3 / 3 | 94.61 Â± 0.23 (+0.82 pp) |
| MT-DTH | 93.82 (+0.03 pp) | 93.55 (-0.24 pp) | 93.86 (+0.07 pp) | 3 / 3 | 93.74 Â± 0.17 (-0.05 pp) |

Δ paper = local accuracy minus the published rounded paper mean, in percentage points (pp), using unrounded local metrics. Per-seed differences compare an individual run with the paper mean; mean differences compare the local mean with the paper mean. No difference is a significance test.

Means and sample SDs use only the completed matched seeds shown in each row. All rows have n=3. No repeated seed is counted as an independent training seed.

| Model | Mean difference from reference over matching seeds (pp) | Δ paper SHD* (93.79%, pp) |
|---|---:|---:|
| DTH | +1.34 | +0.70 |
| LG-DTH | +0.59 | -0.05 |
| DA | +0.18 | -0.46 |
| FP | +0.19 | -0.44 |
| MR (K=2) | +0.74 | +0.10 |
| MT | +1.46 | +0.82 |
| MT-DTH | +0.59 | -0.05 |

Reference differences use unrounded metrics and matching training seeds. Paper differences are each local three-seed mean minus the published, rounded two-layer SHD* mean of 93.79% across 20 runs ([paper results](paper-results.md)); positive values favor the local model. The paper comparison is descriptive, with different sample sizes and no matching paper seeds. MT has the highest observed three-seed mean. Three seeds do not establish statistical significance.

| Run / status | Details |
|---|---|
| DTH | All three completed runs were copied from the other computer. Configs pass runner guards; last checkpoints reach epoch 299 with optimizer state, and validation-selected best checkpoints are present. |
| DA, FP, MT, MR | All 12 replacement runs completed with batch 256 and early stopping disabled. Older batch-512 results and historical mixed histories are excluded. |
| LG-DTH seeds 42/123 | Completed matched baseline runs; retained unchanged. |
| LG-DTH seed 456 fresh rerun | Completed 300/300 epochs; last checkpoint epoch 299, step 7500. Best validation 99.02% at epoch 226; selected checkpoint `epoch=226-step=5675.ckpt`. Final test accuracy 94.43%, loss 0.417353. Batch 256; early stopping disabled. |
| Rerun provenance | Protocol-corrected fresh rerun after viewing the earlier test result. Retain that disclosure; use validation alone for checkpoint selection, then evaluate the official test. Do not choose between repeated seed-456 runs using their test scores. |

| Separate historical result | Test accuracy (%) | Early stopping | Included in matched table |
|---|---:|---|---|
| LG-DTH seed 456 | 93.51 | Enabled; 105 completed epochs | No |

The earlier early-stopping run is preserved at `archive/lg_dth_seed456_early_stopping_20261017/`. Its mixed-stopping three-seed aggregate with seeds 42/123 is **93.43 Â± 0.38%**, descriptive only. The fresh rerun now occupies the canonical seed directory; its verified final test result is included above.

## Spike activity and inference cost

The parameter counts and dense matrix-compute estimates below are recorded in each model's per-seed `TRAINING_SUMMARY.md`; they agree across seeds 42, 123 and 456. All three LG-DTH seeds have now completed; these compute estimates describe recorded validation computation.

| Model | Total parameters | Trainable parameters | Estimated dense GFLOPs / validation sample | Measured-shape dense GFLOPs / test sample |
|---|---:|---:|---:|---:|
| SE-adLIF reference | 450,760 | 450,020 | 0.2619 | 0.2429 |
| DTH | 451,484 | 450,744 | 0.2619 | 0.2429 |
| LG-DTH | 452,924 | 452,184 | 0.2619 | 0.2429 |
| DA | 453,640 | 452,900 | 0.2619 | 0.2429 |
| FP | 450,760 | 450,020 | 0.2619 | 0.2429 |
| MR (K=2) | 450,760 | 450,020 | 0.2619 | 0.2429 |
| MT | 452,200 | 451,460 | 0.2619 | 0.2429 |
| MT-DTH | 452,924 | 452,184 | 0.2619 | 0.2429 |

The existing estimate uses 293.31 average padded validation timesteps/sample and counts feed-forward and recurrent matrix multiply-adds as two FLOPs each. It excludes elementwise neuron updates, including the additional operations of proposed variants, and is not a complete compute or energy estimate. The shared matrix dimensions explain the identical estimates. The new test-shape estimate uses 272.06 average batch-padded test timesteps/sample; the pre-batch stored sequence average is 250.15. Runtime was measured on an NVIDIA GeForce RTX 4090 with PyTorch 2.6.0+cu124, float32 and batch size 256 (final partial batch retained). See the [measurement record and reproducible command](../archive/measurement_provenance/shd_costs_v1/README.md).

| Model | Total hidden spikes/sample | Spikes/sample, L1 / L2 | Firing fraction (%), L1 / L2 | Estimated synaptic operations/sample (M) | Non-matrix logical operations/sample (M), padded | Inference time (ms/sample) |
|---|---|---|---|---|---|---|
| SE-adLIF reference | 10,366 Â± 726 | 8,434 Â± 688 / 1,932 Â± 120 | 9.37 Â± 0.76 / 2.14 Â± 0.13 | 9.76 Â± 0.51 | 5.52 Â± 0.00 | 1.241 Â± 0.011 |
| DTH | 10,514 Â± 613 | 8,320 Â± 239 / 2,194 Â± 434 | 9.24 Â± 0.26 / 2.44 Â± 0.48 | 9.78 Â± 0.30 | 8.46 Â± 0.00 | 1.610 Â± 0.040 |
| LG-DTH | 10,723 Â± 669 | 8,652 Â± 469 / 2,071 Â± 381 | 9.61 Â± 0.52 / 2.30 Â± 0.42 | 9.97 Â± 0.40 | 9.44 Â± 0.00 | 1.901 Â± 0.026 |
| DA | 10,356 Â± 353 | 8,486 Â± 483 / 1,870 Â± 134 | 9.42 Â± 0.54 / 2.08 Â± 0.15 | 9.78 Â± 0.30 | 7.10 Â± 0.00 | 1.542 Â± 0.034 |
| FP | 10,440 Â± 310 | 8,337 Â± 270 / 2,103 Â± 277 | 9.26 Â± 0.30 / 2.34 Â± 0.31 | 9.76 Â± 0.19 | 5.53 Â± 0.00 | 1.734 Â± 0.021 |
| MR (K=2) | 10,431 Â± 271 | 8,239 Â± 522 / 2,191 Â± 301 | 9.15 Â± 0.58 / 2.43 Â± 0.33 | 9.72 Â± 0.27 | 5.42 Â± 0.00 | 1.236 Â± 0.039 |
| MT | 10,637 Â± 609 | 8,406 Â± 556 / 2,232 Â± 88 | 9.33 Â± 0.62 / 2.48 Â± 0.10 | 9.86 Â± 0.42 | 6.70 Â± 0.00 | 1.668 Â± 0.026 |
| MT-DTH | 10,558 Â± 439 | 8,528 Â± 644 / 2,029 Â± 284 | 9.47 Â± 0.72 / 2.25 Â± 0.32 | 9.87 Â± 0.38 | 9.64 Â± 0.00 | 1.991 Â± 0.113 |

Measured on the full official SHD* test set, n=3 training seeds per model; entries are mean Â± sample SD. Original-suite per-seed JSONs, checkpoint hashes, operation breakdowns and timing repeats now live beside each training summary as `COST_RESULTS.json`; see the [measurement layout](../results/valid/MEASUREMENTS.md). Shared frozen source snapshots and exports are retained in `archive/measurement_provenance/shd_costs_v1/`. MT-DTH measurements are saved in each canonical seed directory as `COST_RESULTS.json`; test-batch fingerprints match the original suite. MT-DTH timing was collected in a later session on matching GPU/software; timings remain local benchmarks rather than controlled energy or latency improvements.

Spikes and firing fractions use each stored sequence length before batch padding, including the configured minimum 250 frames and initial ignored frames. This is not the raw event-duration length. Additional batch-padding spikes and costs are retained separately in the JSON records. All models use identical test batches, batch size 256, float32 and validation-selected checkpoints.

Synaptic operations are an ideal dense-fan-out event-driven addition estimate: input frame event multiplicity and hidden spikes times the number of receiving neurons, including the actual previous-step recurrent spikes. Recurrent spikes without a subsequent destination timestep are not counted; zero-valued weights are not pruned. Delay filtering and neuron updates are counted separately. These estimates do not describe the dense GPU execution.

Non-matrix logical operations count the executed cell code on all batch-padded timesteps, including parameter transforms, bias additions, surrogate forward arithmetic, reductions and state/threshold updates. A transcendental, comparison or scalar arithmetic operation each counts as one logical operation; softmax uses an explicit scalar reduction convention. Tensor allocation, indexing, memory movement and matrix multiplication are excluded. These are not hardware FLOPs and must not be added to event-driven SynOps as an energy estimate.

Runtime is synchronized wall time for the uninstrumented `MLPSNN.forward`, after warm-up; data loading/transfer, loss and temporal prediction aggregation are excluded. Hardware, repeat counts and software versions are recorded in JSON. Runtime SD across seeds also reflects system variability; timing was sequential, not an energy benchmark.

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
| DTH | [Per-seed configs, metrics and checkpoints](../results/valid/DTH_SE_adLIF/SHD/) |
| LG-DTH | [Per-seed configs, metrics and checkpoints](../results/not_improved/LG_DTH_SE_adLIF/SHD/) |
| DA | [Per-seed configs, metrics and checkpoints](../results/not_improved/DA_SE_adLIF/SHD/) |
| FP | [Per-seed configs, metrics and checkpoints](../results/not_improved/FP_SE_adLIF/SHD/) |
| MR (K=2) | [Per-seed configs, metrics and checkpoints](../results/not_improved/MR_SE_adLIF/K_2/SHD/) |
| MT | [Per-seed configs, metrics and checkpoints](../results/valid/MT_SE_adLIF/SHD/) |
| MT-DTH | [Per-seed configs, metrics and cost records](../results/not_improved/MT_DTH_SE_adLIF/SHD/) |

[LG-DTH detailed results](../results/not_improved/LG_DTH_SE_adLIF/RESULTS.md) | [Paper versus reference](paper-vs-reference.md)

## Test-set forward operation estimates

| Model | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Matrix + all logical scalar GOp/sample |
|---|---:|---:|---:|
| SE-adLIF reference | 0.242895 Â± 0.000000 | 0.247824 Â± 0.000000 | 0.248414 Â± 0.000000 |
| DTH | 0.242895 Â± 0.000000 | 0.250372 Â± 0.000000 | 0.251353 Â± 0.000000 |
| LG-DTH | 0.242895 Â± 0.000000 | 0.251550 Â± 0.000000 | 0.252338 Â± 0.000000 |
| DA | 0.242895 Â± 0.000000 | 0.249392 Â± 0.000000 | 0.249993 Â± 0.000000 |
| FP | 0.242895 Â± 0.000000 | 0.247837 Â± 0.000000 | 0.248430 Â± 0.000000 |
| MR (K=2) | 0.242895 Â± 0.000000 | 0.247724 Â± 0.000000 | 0.248314 Â± 0.000000 |
| MT | 0.242895 Â± 0.000000 | 0.249006 Â± 0.000000 | 0.249597 Â± 0.000000 |
| MT-DTH | 0.242895 Â± 0.000000 | 0.251553 Â± 0.000000 | 0.252536 Â± 0.000000 |

Mean and sample SD across three seeds, on the same official SHD* test batches. Arithmetic adds counted cell-forward additions, subtractions, multiplications, divisions, negations, reciprocals, bias additions and sum/mean reduction arithmetic to the dense matrix count. Nonlinear functions, powers and comparisons are excluded from the arithmetic estimate; the logical total counts each such scalar operation as one operation. Logical GOp is not hardware GFLOPs. Memory traffic, allocations, indexing, loss and temporal output aggregation are excluded, so these are not exact end-to-end inference FLOP totals or energy measurements. These columns are derived from the saved per-seed operation breakdowns; no SHD* checkpoint was rerun.

## MT-DTH combination follow-up

All three fresh runs completed 300 epochs with batch 256, early stopping disabled and the same split/preprocessing. Test accuracies are lower than both MT and DTH on every matching seed. Mean accuracy is 0.87 percentage points below MT and 0.75 below DTH, while counted neuron-forward arithmetic is higher. Retain this as a negative combination ablation; the lower three-seed SD is not an accuracy improvement. The architecture was chosen after inspecting earlier test results, so retain that development provenance and do not use the new test scores to select further tuning settings.

## Current 27-run grid - 21 October 2026

27/27 accepted completed results: 24 matched and 3 accepted historical ECG reference runs with unmatched stopping/budget. All expected combinations are available locally.

| Model | SHD* accepted | SSC mean-loss accepted | ECG accepted | Total | Matched |
|---|---|---|---|---:|---:|
| Reference | 42, 123, 456 | 42, 123, 456 | 42, 123, 456* | 9/9 | 6/9 |
| DTH | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 9/9 | 9/9 |
| MT | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 | 9/9 | 9/9 |

*Reference ECG is accepted historical evidence; cross-model ECG comparisons remain descriptive.


## SSC mean-loss results

| Model | Seed 42 test (%, Δ paper) | Seed 123 test (%, Δ paper) | Seed 456 test (%, Δ paper) | n | Mean +/- sample SD (%, Δ paper) |
|---|---:|---:|---:|---:|---:|
| Reference | 80.69 (+0.25 pp) | 80.33 (-0.11 pp) | 80.38 (-0.06 pp) | 3 | 80.47 +/- 0.20 (+0.03 pp) |
| DTH | 80.46 (+0.02 pp) | 79.84 (-0.60 pp) | 79.56 (-0.88 pp) | 3 | 79.95 +/- 0.46 (-0.49 pp) |
| MT | 80.01 (-0.43 pp) | 79.98 (-0.46 pp) | 80.23 (-0.21 pp) | 3 | 80.07 +/- 0.14 (-0.37 pp) |

All nine runs completed 40 epochs with batch 256, LR 0.006, mean membrane-potential loss and no early stopping. Checkpoints were selected by validation accuracy. Reference seed 42 retains its diagnostic provenance; the comparison was planned after its test result was observed. Means and sample SD use independent training seeds 42, 123 and 456. No archived summed-loss results are included.

Matching-seed mean test differences from reference, calculated before rounding: DTH -0.512 pp, MT -0.396 pp.

## ECG results - accepted historical reference

| Model | Seed 42 test (%, Δ paper) | Seed 123 test (%, Δ paper) | Seed 456 test (%, Δ paper) | n | Mean +/- sample SD (%, Δ paper) |
|---|---:|---:|---:|---:|---:|
| Reference* | 88.13 (-0.05 pp) | 88.72 (+0.54 pp) | 88.11 (-0.07 pp) | 3 | 88.32 +/- 0.35 (+0.14 pp) |
| DTH | 87.81 (-0.37 pp) | 87.81 (-0.37 pp) | 87.27 (-0.91 pp) | 3 | 87.63 +/- 0.31 (-0.55 pp) |
| MT | 87.26 (-0.92 pp) | 88.19 (+0.01 pp) | 87.98 (-0.20 pp) | 3 | 87.81 +/- 0.49 (-0.37 pp) |

*Reference seed 42 completed 383 epochs; seeds 123/456 enabled early stopping and completed 283/185 epochs. DTH/MT all completed 400 epochs with early stopping disabled. Validation fraction 0.05 and split seed 42 are shared. All three model rows are accepted evidence, but cross-model ECG differences are descriptive because budgets/stopping differ. No reference retraining is required under this reporting agreement. Original configs and metadata remain unchanged.

## Historical SSC summed-loss evidence - excluded from current mean-loss grid

| Model | Seed 42 | Seed 123 | Seed 456 | Accepted metric completions | Mean +/- sample SD (%) |
|---|---:|---:|---:|---:|---:|
| SE-adLIF reference | 78.33 | 78.12 | 78.17 | 3 | 78.21 +/- 0.11 |
| DTH | 78.22 | 78.31 | 78.04 | 3 | 78.19 +/- 0.14 |
| MT | 77.60 | 78.14 | 77.46 | 3 | 77.73 +/- 0.36 |
| DA | 77.88 | pending | pending | 1 | 77.88 (n=1) |
| FP | 77.67 (unverified) | pending | pending | 0 | excluded |
| MR (K=2) | 76.93 | pending | pending | 1 | 76.93 (n=1) |
| LG-DTH | pending | pending | pending | 0 | pending |
| MT-DTH | pending | pending | pending | 0 | pending |

These original results are preserved under `archive/ssc_summed_loss_20261019/` for reference/DTH/MT. They are not extra independent runs in the 27-run grid. Other retained models keep their own original protocols. FP SSC has conflicting validation history and remains unverified; its displayed score is historical only.

See the [per-seed grid and evidence links](../results/valid/COMPARISON_STATUS.md), [SSC comparison](../results/valid/SSC_MEAN_COMPARISON.md), [accepted ECG reference](../results/valid/reference/ECG/RESULTS.md) and [20 October audit](../docs/audits/comparison-grid-2026-10-20.md). Per-model reports retain per-seed activity/cost records; no measurements were rerun for this update.
