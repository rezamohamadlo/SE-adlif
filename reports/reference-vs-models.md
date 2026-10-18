# Reference versus proposed models - SHD*

Snapshot: 18 October 2026. **8 models x 3 training seeds = 24 completed SHD* runs**, including the MT-DTH combination follow-up. The original seven-model suite is retained; MT-DTH was proposed after observing its results and is a disclosed follow-up, not a preregistered independent architecture comparison. Completed early-stopping results are kept separately below.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | Matched completed / 3 | Mean ± sample SD (%) |
|---|---:|---:|---:|---:|---:|
| SE-adLIF reference | 92.93 | 92.89 | 93.64 | 3 / 3 | 93.15 ± 0.42 |
| DTH | 94.70 | 94.88 | 93.90 | 3 / 3 | 94.49 ± 0.52 |
| LG-DTH | 93.77 | 93.02 | 94.43 | 3 / 3 | 93.74 ± 0.71 |
| DA | 93.73 | 93.99 | 92.27 | 3 / 3 | 93.33 ± 0.93 |
| FP | 93.51 | 92.49 | 94.04 | 3 / 3 | 93.35 ± 0.79 |
| MR (K=2) | 94.04 | 93.95 | 93.68 | 3 / 3 | 93.89 ± 0.18 |
| MT | 94.88 | 94.52 | 94.43 | 3 / 3 | 94.61 ± 0.23 |
| MT-DTH | 93.82 | 93.55 | 93.86 | 3 / 3 | 93.74 ± 0.17 |

Means and sample SDs use only the completed matched seeds shown in each row. All rows have n=3. No repeated seed is counted as an independent training seed.

| Model | Mean difference from reference over matching seeds (pp) |
|---|---:|
| DTH | +1.34 |
| LG-DTH | +0.59 |
| DA | +0.18 |
| FP | +0.19 |
| MR (K=2) | +0.74 |
| MT | +1.46 |
| MT-DTH | +0.59 |

Differences use unrounded metrics and matching training seeds. MT has the highest observed three-seed mean. Three seeds do not establish statistical significance.

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

The earlier early-stopping run is preserved at `results/archive/lg_dth_seed456_early_stopping_20261017/`. Its mixed-stopping three-seed aggregate with seeds 42/123 is **93.43 ± 0.38%**, descriptive only. The fresh rerun now occupies the canonical seed directory; its verified final test result is included above.

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

The existing estimate uses 293.31 average padded validation timesteps/sample and counts feed-forward and recurrent matrix multiply-adds as two FLOPs each. It excludes elementwise neuron updates, including the additional operations of proposed variants, and is not a complete compute or energy estimate. The shared matrix dimensions explain the identical estimates. The new test-shape estimate uses 272.06 average batch-padded test timesteps/sample; the pre-batch stored sequence average is 250.15. Runtime was measured on an NVIDIA GeForce RTX 4090 with PyTorch 2.6.0+cu124, float32 and batch size 256 (final partial batch retained). See the [measurement record and reproducible command](../results/measurements/shd_costs_v1/README.md).

| Model | Total hidden spikes/sample | Spikes/sample, L1 / L2 | Firing fraction (%), L1 / L2 | Estimated synaptic operations/sample (M) | Non-matrix logical operations/sample (M), padded | Inference time (ms/sample) |
|---|---|---|---|---|---|---|
| SE-adLIF reference | 10,366 ± 726 | 8,434 ± 688 / 1,932 ± 120 | 9.37 ± 0.76 / 2.14 ± 0.13 | 9.76 ± 0.51 | 5.52 ± 0.00 | 1.241 ± 0.011 |
| DTH | 10,514 ± 613 | 8,320 ± 239 / 2,194 ± 434 | 9.24 ± 0.26 / 2.44 ± 0.48 | 9.78 ± 0.30 | 8.46 ± 0.00 | 1.610 ± 0.040 |
| LG-DTH | 10,723 ± 669 | 8,652 ± 469 / 2,071 ± 381 | 9.61 ± 0.52 / 2.30 ± 0.42 | 9.97 ± 0.40 | 9.44 ± 0.00 | 1.901 ± 0.026 |
| DA | 10,356 ± 353 | 8,486 ± 483 / 1,870 ± 134 | 9.42 ± 0.54 / 2.08 ± 0.15 | 9.78 ± 0.30 | 7.10 ± 0.00 | 1.542 ± 0.034 |
| FP | 10,440 ± 310 | 8,337 ± 270 / 2,103 ± 277 | 9.26 ± 0.30 / 2.34 ± 0.31 | 9.76 ± 0.19 | 5.53 ± 0.00 | 1.734 ± 0.021 |
| MR (K=2) | 10,431 ± 271 | 8,239 ± 522 / 2,191 ± 301 | 9.15 ± 0.58 / 2.43 ± 0.33 | 9.72 ± 0.27 | 5.42 ± 0.00 | 1.236 ± 0.039 |
| MT | 10,637 ± 609 | 8,406 ± 556 / 2,232 ± 88 | 9.33 ± 0.62 / 2.48 ± 0.10 | 9.86 ± 0.42 | 6.70 ± 0.00 | 1.668 ± 0.026 |
| MT-DTH | 10,558 ± 439 | 8,528 ± 644 / 2,029 ± 284 | 9.47 ± 0.72 / 2.25 ± 0.32 | 9.87 ± 0.38 | 9.64 ± 0.00 | 1.991 ± 0.113 |

Measured on the full official SHD* test set, n=3 training seeds per model; entries are mean ± sample SD. Original-suite per-seed JSONs, checkpoint hashes, operation breakdowns and timing repeats: `results/measurements/shd_costs_v1/`. MT-DTH measurements are saved in each canonical seed directory as `COST_RESULTS.json`; test-batch fingerprints match the original suite. MT-DTH timing was collected in a later session on matching GPU/software; timings remain local benchmarks rather than controlled energy or latency improvements.

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
| DTH | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/DTH_SE_adLIF/SHD/) |
| LG-DTH | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/) |
| DA | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/DA_SE_adLIF/SHD/) |
| FP | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/FP_SE_adLIF/SHD/) |
| MR (K=2) | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/MR_SE_adLIF/K_2/SHD/) |
| MT | [Per-seed configs, metrics and checkpoints](../results/valid/protocol_v2/MT_SE_adLIF/SHD/) |
| MT-DTH | [Per-seed configs, metrics and cost records](../results/valid/protocol_v2/MT_DTH_SE_adLIF/SHD/) |

[LG-DTH detailed results](../results/valid/protocol_v2/LG_DTH_SE_adLIF/RESULTS.md) | [Paper versus reference](paper-vs-reference.md)

## Test-set forward operation estimates

| Model | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Matrix + all logical scalar GOp/sample |
|---|---:|---:|---:|
| SE-adLIF reference | 0.242895 ± 0.000000 | 0.247824 ± 0.000000 | 0.248414 ± 0.000000 |
| DTH | 0.242895 ± 0.000000 | 0.250372 ± 0.000000 | 0.251353 ± 0.000000 |
| LG-DTH | 0.242895 ± 0.000000 | 0.251550 ± 0.000000 | 0.252338 ± 0.000000 |
| DA | 0.242895 ± 0.000000 | 0.249392 ± 0.000000 | 0.249993 ± 0.000000 |
| FP | 0.242895 ± 0.000000 | 0.247837 ± 0.000000 | 0.248430 ± 0.000000 |
| MR (K=2) | 0.242895 ± 0.000000 | 0.247724 ± 0.000000 | 0.248314 ± 0.000000 |
| MT | 0.242895 ± 0.000000 | 0.249006 ± 0.000000 | 0.249597 ± 0.000000 |
| MT-DTH | 0.242895 ± 0.000000 | 0.251553 ± 0.000000 | 0.252536 ± 0.000000 |

Mean and sample SD across three seeds, on the same official SHD* test batches. Arithmetic adds counted cell-forward additions, subtractions, multiplications, divisions, negations, reciprocals, bias additions and sum/mean reduction arithmetic to the dense matrix count. Nonlinear functions, powers and comparisons are excluded from the arithmetic estimate; the logical total counts each such scalar operation as one operation. Logical GOp is not hardware GFLOPs. Memory traffic, allocations, indexing, loss and temporal output aggregation are excluded, so these are not exact end-to-end inference FLOP totals or energy measurements. These columns are derived from the saved per-seed operation breakdowns; no SHD* checkpoint was rerun.

## MT-DTH combination follow-up

All three fresh runs completed 300 epochs with batch 256, early stopping disabled and the same split/preprocessing. Test accuracies are lower than both MT and DTH on every matching seed. Mean accuracy is 0.87 percentage points below MT and 0.75 below DTH, while counted neuron-forward arithmetic is higher. Retain this as a negative combination ablation; the lower three-seed SD is not an accuracy improvement. The architecture was chosen after inspecting earlier test results, so retain that development provenance and do not use the new test scores to select further tuning settings.
