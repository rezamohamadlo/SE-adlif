# Paper and proposed-model comparison

## SHD* speaker holdout + augmentation versus shuffled reference

This descriptive comparison adds the imported `shd_speaker_augmented_v1`
speaker-holdout experiment to the canonical shuffled SHD* runs and the local
SE-adLIF reference. Train and validation values are taken at each
validation-selected checkpoint; test accuracy is evaluated afterward. Values
are percentages, with seeds shown in the order 42 / 123 / 456.

| Model / protocol | Train by seed | Train mean +/- sample SD | Validation by seed | Validation mean +/- sample SD | Test by seed | Test mean +/- sample SD (Δ paper) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Local reference, shuffled | 99.89 / 100.00 / 100.00 | 99.96 +/- 0.06 | 99.14 / 99.39 / 99.08 | 99.20 +/- 0.16 | 92.93 / 92.89 / 93.64 | 93.15 +/- 0.42 (-0.64 pp) |
| DTH, shuffled | 100.00 / 100.00 / 100.00 | 100.00 +/- 0.00 | 99.14 / 99.08 / 99.02 | 99.08 +/- 0.06 | 94.70 / 94.88 / 93.90 | 94.49 +/- 0.52 (+0.70 pp) |
| DTH, speaker holdout + augmentation | 99.55 / 99.89 / 99.83 | 99.76 +/- 0.18 | 90.03 / 90.03 / 91.62 | 90.56 +/- 0.92 | 92.84 / 93.07 / 93.51 | 93.14 +/- 0.34 (-0.65 pp) |
| LG-DTH, shuffled | 100.00 / 100.00 / 100.00 | 100.00 +/- 0.00 | 99.14 / 99.26 / 99.02 | 99.14 +/- 0.12 | 93.77 / 93.02 / 94.43 | 93.74 +/- 0.71 (-0.05 pp) |
| LG-DTH, speaker holdout + augmentation | 99.70 / 99.55 / 99.58 | 99.61 +/- 0.08 | 91.19 / 92.35 / 91.74 | 91.76 +/- 0.58 | 93.68 / 96.64 / 92.45 | 94.26 +/- 2.16 (+0.47 pp) |
| Paper SE-adLIF (20 runs) | Not reported | Not reported | Not reported | Not reported | Not reported by seed | 93.79 +/- 0.76 |

Delta paper is the local test mean minus the published SHD* paper mean of
93.79%, in percentage points. Relative to the matching shuffled protocol, the
speaker-holdout test mean changes by -1.35 pp for DTH and +0.52 pp for LG-DTH.
The speaker split uses validation speakers 0 and 3, so these results are not a
matched replacement for the random-recording SHD* comparison. The paper does
not report train or validation accuracy. Sources: [speaker-holdout results](../results/experiments/shd_speaker_augmented_v1/README.md), [local reference](../results/valid/reference/SHD/RESULTS.md), [shuffled DTH](../results/valid/DTH_SE_adLIF/RESULTS.md), [shuffled LG-DTH](../results/not_improved/LG_DTH_SE_adLIF/RESULTS.md), and [paper results](paper-results.md).

## Test accuracy

### Without TTA

| Dataset | Paper | Reference (Δ paper) | DTH (Δ paper) | MT (Δ paper) | AMT (Δ paper) | MT-TC (Δ paper) | LT_DTH (Δ paper) | SAMT (Δ paper) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | 93.79 ± 0.76 | 93.15 ± 0.42 (-0.64 pp) | **94.49 ± 0.52** (+0.70 pp) | **94.61 ± 0.23** (+0.82 pp) | 92.71 ± 1.94 (-1.08 pp) | **93.80 ± 0.95** (+0.01 pp) | **93.98 ± 0.63** (+0.19 pp) | 93.73 ± 0.52 (-0.06 pp) |
| SSC | 80.44 ± 0.26 | **80.47 ± 0.20** (+0.03 pp) | 79.95 ± 0.46 (-0.49 pp) | 80.07 ± 0.14 (-0.37 pp) | 80.26 ± 0.18 (-0.18 pp) | 80.14 ± 0.11 (-0.30 pp) | 80.17 ± 0.19 (-0.27 pp) | 80.10 ± 0.25 (-0.34 pp) |
| ECG | 88.18 ± 0.36 | **88.32 ± 0.35** (+0.14 pp) | 87.63 ± 0.31 (-0.55 pp) | 87.81 ± 0.49 (-0.37 pp) | 87.83 ± 0.18 (-0.35 pp) | 88.13 ± 0.44 (-0.05 pp) | 87.67 ± 0.61 (-0.51 pp) | **88.39 ± 0.22** (+0.21 pp) |

Snapshot: 5 November 2026. Columns: Dataset, Paper, Reference, DTH, MT, AMT, MT-TC, LT_DTH and SAMT. Reference = SE-adLIF; MT-TC = MT with temporal-consistency regularization; LT_DTH = learned-trace DTH; SAMT = SA-MT.

Main comparisons use **ordinary inference without test-time augmentation (TTA)** for accuracy and cost. Accuracy (%) is mean ± sample SD across matching training seeds 42, 123 and 456 (n=3), using final test evaluation of validation-selected checkpoints. **Bold** marks a local mean above the paper mean. Δ paper = local mean minus the published paper mean, in percentage points (pp), calculated before rounding. Paper sample sizes are SHD*/ECG n=20 and SSC n=5. These descriptive comparisons do not establish statistical significance. Exploratory TTA accuracy is presented separately below and does not determine the main reported result.

The six proposed models share dataset-specific baseline optimization, preprocessing and budget. MT-TC adds the opt-in `mt_temporal_consistency_v1` training objective (weight 0.1, window 5); this intentional objective deviation is not identical to ordinary MT training. Reference SHD*/SSC uses the current accepted reference records. Reference ECG is accepted historical evidence with unmatched stopping/budget (early stopping enabled for seeds 123/456), descriptive only; see [reference provenance](paper-vs-reference.md). SSC uses `ssc_mean_loss_v1`. SHD* uses a fixed 20% training-derived validation split with split seed 42.

### With TTA (exploratory)

| Dataset | Paper | Reference (Δ paper) | DTH (Δ paper) | MT (Δ paper) | AMT (Δ paper) | MT-TC (Δ paper) | LT_DTH (Δ paper) | SAMT (Δ paper) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | 93.79 ± 0.76 | Not evaluated | **94.64 ± 0.54** (+0.85 pp) | **94.61 ± 0.31** (+0.82 pp) | 92.76 ± 2.11 (-1.03 pp) | **93.80 ± 0.91** (+0.01 pp) | **93.93 ± 0.71** (+0.14 pp) | 93.77 ± 0.61 (-0.02 pp) |
| SSC | 80.44 ± 0.26 | Not evaluated | 80.34 ± 0.49 (-0.10 pp) | 80.39 ± 0.01 (-0.05 pp) | **80.72 ± 0.28** (+0.28 pp) | **80.47 ± 0.10** (+0.03 pp) | **80.60 ± 0.31** (+0.16 pp) | **80.54 ± 0.22** (+0.10 pp) |
| ECG | 88.18 ± 0.36 | Not evaluated | **88.40 ± 0.17** (+0.22 pp) | **88.53 ± 0.54** (+0.35 pp) | **88.60 ± 0.17** (+0.42 pp) | **88.80 ± 0.41** (+0.62 pp) | **88.42 ± 0.61** (+0.24 pp) | **89.06 ± 0.28** (+0.88 pp) |

**TTA accuracy change (percentage points)**

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | Not evaluated | +0.15 | +0.00 | +0.04 | +0.00 | -0.04 | +0.04 |
| SSC | Not reported | Not evaluated | +0.38 | +0.32 | +0.46 | +0.33 | +0.43 | +0.44 |
| ECG | Not reported | Not evaluated | +0.77 | +0.72 | +0.77 | +0.67 | +0.74 | +0.66 |

Change = accuracy with TTA minus accuracy without TTA, averaged over paired seeds 42, 123 and 456 (n=3), using unrounded saved test results. Positive values favor TTA; these exploratory differences do not establish statistical significance.

TTA uses frozen validation-selected checkpoints and averages class probabilities over input gains 1.0, 0.95 and 1.05 (three forward passes). Values are mean ± sample SD across training seeds 42, 123 and 456 (n=3); Δ paper uses the unrounded TTA mean. All numeric local entries in this table use TTA, including when ordinary inference has a higher score. Reference TTA has not been evaluated and is not inferred from ordinary inference.

These evaluations are exploratory because the test results were previously inspected; they do not establish independent improvements or statistical significance. The cost tables below describe ordinary inference and must not be interpreted as TTA latency or energy.

Sources: [DTH SHD*/ECG](../results/experiments/dth_tta_v1/RESULTS.md), [DTH SSC](../results/experiments/dth_tta_v2/RESULTS.md), [MT SHD*/ECG](../results/experiments/mt_tta_v1/RESULTS.md), [MT SSC](../results/experiments/mt_tta_v2/RESULTS.md), [AMT](../results/experiments/amt_tta_v1/RESULTS.md), [SAMT](../results/experiments/sa_mt_tta_v1/RESULTS.md), [MT-TC](../results/experiments/mt_tc_tta_v1/RESULTS.md), [LT_DTH](../results/experiments/lt_dth_tta_v1/RESULTS.md).

## Three versus ten training seeds

Ordinary-inference final test accuracy (%), mean ± sample SD. The three-seed group uses 42, 123 and 456; the ten-seed group contains those same three plus 1009, 2027, 3037, 4049, 5051, 6067 and 7079. All ten distinct seeds have final-test records for each group below.

| Dataset | Model | Paper | 3 seeds (Δ paper) | 10 seeds (Δ paper) | Change: 10 minus 3 (pp) |
| --- | --- | ---: | ---: | ---: | ---: |
| SHD* | DTH | 93.79 ± 0.76 | 94.49 ± 0.52 (+0.70 pp) | 94.10 ± 0.56 (+0.31 pp) | -0.39 |
| SHD* | MT | 93.79 ± 0.76 | 94.61 ± 0.23 (+0.82 pp) | 93.89 ± 1.13 (+0.10 pp) | -0.72 |
| ECG | MT | 88.18 ± 0.36 | 87.81 ± 0.49 (-0.37 pp) | 87.69 ± 0.45 (-0.49 pp) | -0.12 |
| ECG | SAMT | 88.18 ± 0.36 | 88.39 ± 0.22 (+0.21 pp) | 87.94 ± 0.71 (-0.24 pp) | -0.46 |

Δ paper follows the definition above. The last column compares the two local means using unrounded metrics. These groups overlap, so the difference is descriptive rather than a comparison of independent samples; sample SD measures variation across training seeds, not uncertainty of the mean. A smaller SD or a higher mean is not guaranteed by adding seeds.

The seven extension runs were trained with validation-only selection and subsequently tested separately using the saved validation-selected checkpoints. Their saved test records are distinct from validation scores. Training/preprocessing settings were checked against the original model's saved configuration; split seed remains 42. Existing test scores had already been observed. Frozen evaluation plans originally reported the seven additional seeds separately; this combined ten-seed summary is a descriptive aggregation requested afterward, with no seed exclusions, new checkpoint selection, training or inference. It does not establish independent confirmation or statistical significance.

Extension official-test sources (original-three sources appear in the per-seed accuracy table below):

| Dataset | Model | Additional-seed test records |
| --- | --- | --- |
| SHD* | DTH | [1009](../results/experiments/extended_seeds_v1/test_remaining_v1/dth_shd/seed_1009/TEST_RESULTS.json), [2027](../results/experiments/extended_seeds_v1/test_remaining_v1/dth_shd/seed_2027/TEST_RESULTS.json), [3037](../results/experiments/extended_seeds_v1/test_remaining_v1/dth_shd/seed_3037/TEST_RESULTS.json), [4049](../results/experiments/extended_seeds_v1/test_remaining_v1/dth_shd/seed_4049/TEST_RESULTS.json), [5051](../results/experiments/extended_seeds_v1/test_remaining_v1/dth_shd/seed_5051/TEST_RESULTS.json), [6067](../results/experiments/extended_seeds_v1/test_remaining_v1/dth_shd/seed_6067/TEST_RESULTS.json), [7079](../results/experiments/extended_seeds_v1/test_remaining_v1/dth_shd/seed_7079/TEST_RESULTS.json) |
| SHD* | MT | [1009](../results/experiments/extended_seeds_v1/test_mt_shd_v1/seed_1009/TEST_RESULTS.json), [2027](../results/experiments/extended_seeds_v1/test_mt_shd_v1/seed_2027/TEST_RESULTS.json), [3037](../results/experiments/extended_seeds_v1/test_mt_shd_v1/seed_3037/TEST_RESULTS.json), [4049](../results/experiments/extended_seeds_v1/test_mt_shd_v1/seed_4049/TEST_RESULTS.json), [5051](../results/experiments/extended_seeds_v1/test_mt_shd_v1/seed_5051/TEST_RESULTS.json), [6067](../results/experiments/extended_seeds_v1/test_mt_shd_v1/seed_6067/TEST_RESULTS.json), [7079](../results/experiments/extended_seeds_v1/test_mt_shd_v1/seed_7079/TEST_RESULTS.json) |
| ECG | MT | [1009](../results/experiments/extended_seeds_v1/pc1_ecg/MT_SE_adLIF/ECG/seed_1009/TEST_RESULTS.json), [2027](../results/experiments/extended_seeds_v1/pc1_ecg/MT_SE_adLIF/ECG/seed_2027/TEST_RESULTS.json), [3037](../results/experiments/extended_seeds_v1/pc1_ecg/MT_SE_adLIF/ECG/seed_3037/TEST_RESULTS.json), [4049](../results/experiments/extended_seeds_v1/pc1_ecg/MT_SE_adLIF/ECG/seed_4049/TEST_RESULTS.json), [5051](../results/experiments/extended_seeds_v1/pc1_ecg/MT_SE_adLIF/ECG/seed_5051/TEST_RESULTS.json), [6067](../results/experiments/extended_seeds_v1/pc1_ecg/MT_SE_adLIF/ECG/seed_6067/TEST_RESULTS.json), [7079](../results/experiments/extended_seeds_v1/pc1_ecg/MT_SE_adLIF/ECG/seed_7079/TEST_RESULTS.json) |
| ECG | SAMT | [1009](../results/experiments/extended_seeds_v1/test_remaining_v1/sa_mt_ecg/seed_1009/TEST_RESULTS.json), [2027](../results/experiments/extended_seeds_v1/test_remaining_v1/sa_mt_ecg/seed_2027/TEST_RESULTS.json), [3037](../results/experiments/extended_seeds_v1/test_remaining_v1/sa_mt_ecg/seed_3037/TEST_RESULTS.json), [4049](../results/experiments/extended_seeds_v1/test_remaining_v1/sa_mt_ecg/seed_4049/TEST_RESULTS.json), [5051](../results/experiments/extended_seeds_v1/test_remaining_v1/sa_mt_ecg/seed_5051/TEST_RESULTS.json), [6067](../results/experiments/extended_seeds_v1/test_remaining_v1/sa_mt_ecg/seed_6067/TEST_RESULTS.json), [7079](../results/experiments/extended_seeds_v1/test_remaining_v1/sa_mt_ecg/seed_7079/TEST_RESULTS.json) |

Frozen plans: [MT SHD*](../results/experiments/extended_seeds_v1/test_mt_shd_v1/FROZEN_TEST_PLAN.json), [DTH SHD* and SAMT ECG](../results/experiments/extended_seeds_v1/test_remaining_v1/FROZEN_TEST_PLAN.json). MT ECG uses its separately saved per-seed `TEST_RESULTS.json` records linked above.

## Inference time

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 1.24 ± 0.01 | 1.61 ± 0.04 | 1.67 ± 0.03 | 1.84 ± 0.12 | 1.54 ± 0.08 | 1.79 ± 0.01 | 1.59 ± 0.01 |
| SSC | Not reported | 1.41 ± 0.07 | 1.74 ± 0.16 | 1.70 ± 0.14 | 2.02 ± 0.11 | 1.83 ± 0.11 | 1.91 ± 0.01 | 1.76 ± 0.02 |
| ECG | Not reported | 33.87 ± 4.54 | 43.91 ± 0.31 | 44.67 ± 0.53 | 45.96 ± 1.06 | 46.34 ± 0.49 | 45.91 ± 1.14 | 42.43 ± 1.12 |

All cost tables use ordinary inference without TTA. Timing is forward-only milliseconds/sample, mean ± sample SD across three seeds.

Saved timing uses three repeats, two warm-up batches, batch sizes 256 for SHD*/SSC and 64 for ECG, RTX 4090 and high matmul precision. It excludes loading, transfer, loss and temporal prediction aggregation. Sessions and evaluator environments differ; these are descriptive measurements, not a controlled hardware speed benchmark.

## Parameters, activity and computational cost

### Total parameters

| Dataset | Paper | Reference (Δ paper) | DTH (Δ paper) | MT (Δ paper) | AMT (Δ paper) | MT-TC (Δ paper) | LT_DTH (Δ paper) | SAMT (Δ paper) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | 0.45M | 450,760 (≈+760 params) | 451,484 (≈+1,484 params) | 452,200 (≈+2,200 params) | 452,920 (≈+2,920 params) | 452,200 (≈+2,200 params) | 451,488 (≈+1,488 params) | 452,920 (≈+2,920 params) |
| SSC | 1.6M | 1,689,910 (≈+89,910 params) | 1,691,354 (≈+91,354 params) | 1,692,790 (≈+92,790 params) | 1,694,230 (≈+94,230 params) | 1,692,790 (≈+92,790 params) | 1,691,358 (≈+91,358 params) | 1,694,230 (≈+94,230 params) |
| ECG | 4.6k | 4,692 (≈+92 params) | 4,768 (≈+168 params) | 4,836 (≈+236 params) | 4,908 (≈+308 params) | 4,836 (≈+236 params) | 4,772 (≈+172 params) | 4,908 (≈+308 params) |

Published counts are rounded: 0.45M = 450,000, 1.6M = 1,600,000 and 4.6k = 4,600. Count differences from those rounded baselines are approximate (≈), in parameters.

All statistics below come from the same per-seed cost records. Parameter counts are exact and identical across seeds. Other values are mean ± sample SD (n=3); zero SD means equal counted costs, not measurement certainty.

### Trainable parameters

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 450,020 | 450,744 | 451,460 | 452,180 | 451,460 | 450,748 | 452,180 |
| SSC | Not reported | 1,688,435 | 1,689,879 | 1,691,315 | 1,692,755 | 1,691,315 | 1,689,883 | 1,692,755 |
| ECG | Not reported | 4,614 | 4,690 | 4,758 | 4,830 | 4,758 | 4,694 | 4,830 |



### Hidden spikes per sample

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 10,365.8 ± 726.0 | 10,513.9 ± 612.6 | 10,637.4 ± 608.6 | 10,739.6 ± 375.6 | 10,628.1 ± 514.5 | 10,220.3 ± 502.9 | 10,712.1 ± 587.9 |
| SSC | Not reported | 45,301.2 ± 1,640.2 | 45,601.5 ± 4,030.9 | 49,500.0 ± 580.7 | 49,383.4 ± 2,491.1 | 48,358.8 ± 2,136.6 | 43,980.5 ± 1,696.4 | 49,151.0 ± 1,644.4 |
| ECG | Not reported | 26,000.4 ± 2,012.7 | 30,057.7 ± 2,822.3 | 26,665.3 ± 378.0 | 27,536.2 ± 386.6 | 28,281.4 ± 989.2 | 29,788.9 ± 1,455.6 | 27,166.5 ± 950.0 |



### Event-driven synaptic-operation estimate (million SynOps/sample)

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 9.76 ± 0.51 | 9.78 ± 0.30 | 9.86 ± 0.42 | 9.98 ± 0.12 | 9.88 ± 0.34 | 9.71 ± 0.39 | 9.87 ± 0.37 |
| SSC | Not reported | 53.34 ± 1.69 | 53.51 ± 4.73 | 58.35 ± 0.32 | 58.29 ± 3.00 | 57.23 ± 2.56 | 52.34 ± 1.80 | 58.17 ± 2.31 |
| ECG | Not reported | 1.40 ± 0.13 | 1.61 ± 0.17 | 1.42 ± 0.04 | 1.48 ± 0.03 | 1.53 ± 0.06 | 1.61 ± 0.13 | 1.45 ± 0.05 |



### Hidden spike sparsity (%)

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 90.63 ± 0.76 / 97.86 ± 0.13 | 90.76 ± 0.26 / 97.56 ± 0.48 | 90.67 ± 0.62 / 97.52 ± 0.10 | 90.39 ± 0.11 / 97.68 ± 0.50 | 90.57 ± 0.52 / 97.62 ± 0.33 | 90.62 ± 0.64 / 98.03 ± 0.08 | 90.70 ± 0.47 / 97.40 ± 0.20 |
| SSC | Not reported | 91.14 ± 0.32 / 87.88 ± 0.47 | 91.18 ± 1.14 / 87.71 ± 0.72 | 89.89 ± 0.12 / 87.20 ± 0.37 | 89.87 ± 0.77 / 87.27 ± 0.40 | 90.06 ± 0.68 / 87.55 ± 0.47 | 91.15 ± 0.35 / 88.49 ± 0.43 | 89.83 ± 0.73 / 87.42 ± 0.16 |
| ECG | Not reported | 78.30 ± 2.92 / 66.19 ± 1.38 | 75.17 ± 3.99 / 60.66 ± 2.86 | 78.77 ± 2.62 / 64.30 ± 2.49 | 77.25 ± 1.12 / 63.96 ± 0.78 | 75.31 ± 1.50 / 64.31 ± 1.32 | 74.61 ± 5.07 / 61.79 ± 1.97 | 77.69 ± 1.22 / 64.31 ± 2.18 |

Each cell reports L1 / L2. Sparsity is 100 minus the saved firing percentage over stored sample timesteps, including minimum padding and burn-in but excluding added batch padding.

### Dense matrix cost (GFLOPs/sample)

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 |
| SSC | Not reported | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 |
| ECG | Not reported | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 |



### Matrix plus scalar arithmetic cost (GFLOPs/sample)

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 0.247824 ± 0.000000 | 0.250372 ± 0.000000 | 0.249006 ± 0.000000 | 0.250769 ± 0.000000 | 0.249006 ± 0.000000 | 0.250372 ± 0.000000 | 0.249789 ± 0.000000 |
| SSC | Not reported | 1.019584 ± 0.000000 | 1.025202 ± 0.000000 | 1.022190 ± 0.000000 | 1.026077 ± 0.000000 | 1.022190 ± 0.000000 | 1.025202 ± 0.000000 | 1.023918 ± 0.000000 |
| ECG | Not reported | 0.013455 ± 0.000000 | 0.014678 ± 0.000000 | 0.014033 ± 0.000000 | 0.014879 ± 0.000000 | 0.014033 ± 0.000000 | 0.014678 ± 0.000000 | 0.014407 ± 0.000000 |



### Logical operation estimate (GOp/sample)

| Dataset | Paper | Reference | DTH | MT | AMT | MT-TC | LT_DTH | SAMT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 0.248414 ± 0.000000 | 0.251353 ± 0.000000 | 0.249597 ± 0.000000 | 0.251555 ± 0.000000 | 0.249597 ± 0.000000 | 0.251353 ± 0.000000 | 0.250380 ± 0.000000 |
| SSC | Not reported | 1.020884 ± 0.000000 | 1.027365 ± 0.000000 | 1.023493 ± 0.000000 | 1.027811 ± 0.000000 | 1.023493 ± 0.000000 | 1.027365 ± 0.000000 | 1.025221 ± 0.000000 |
| ECG | Not reported | 0.013740 ± 0.000000 | 0.015150 ± 0.000000 | 0.014322 ± 0.000000 | 0.015260 ± 0.000000 | 0.014322 ± 0.000000 | 0.015151 ± 0.000000 | 0.014696 ± 0.000000 |

Hidden spikes sum both hidden layers and exclude output activity. SynOps estimate ideal event-driven fan-out additions; they do not measure dense GPU operations or energy. ECG analog-input projection is excluded from SynOps and counted separately as dense MACs in source records.

Dense matrix cost counts each multiply-add as two FLOPs. Arithmetic adds counted neuron scalar arithmetic, excluding nonlinearities and comparisons. Logical GOp additionally includes counted nonlinear/comparison operations and is not hardware GFLOPs. Compute uses executed batch-padded sequence extent and excludes memory traffic, loading, transfer, loss and temporal aggregation. Paper classification activity/compute values are unavailable; rounded parameter counts come from Table 2 (PDF page 11), which does not separate trainable counts. Audio-bottleneck spike rates are not comparable classification activity.

## Per-seed measurements and sources

| Dataset | Model | Per-seed ms/sample | Cost records by seed |
|---|---:|---:|---:|
| SHD* | Reference | 1.23 / 1.24 / 1.25 | [42](../results/valid/reference/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/reference/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/reference/SHD/seed_456/COST_RESULTS.json) |
| SHD* | DTH | 1.66 / 1.60 / 1.58 | [42](../results/valid/DTH_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | MT | 1.64 / 1.68 / 1.68 | [42](../results/valid/MT_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | AMT | 1.98 / 1.77 / 1.78 | [42](../results/not_improved/AMT_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/not_improved/AMT_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/not_improved/AMT_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | MT-TC | 1.63 / 1.51 / 1.49 | [42](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | LT_DTH | 1.78 / 1.79 / 1.79 | [42](../results/not_improved/LT_DTH_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/not_improved/LT_DTH_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/not_improved/LT_DTH_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | SAMT | 1.59 / 1.58 / 1.59 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SSC | Reference | 1.32 / 1.46 / 1.44 | [42](../results/valid/reference/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/reference/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/reference/SSC/seed_456/COST_RESULTS.json) |
| SSC | DTH | 1.61 / 1.92 / 1.70 | [42](../results/valid/DTH_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | MT | 1.56 / 1.85 / 1.69 | [42](../results/valid/MT_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | AMT | 1.98 / 1.94 / 2.15 | [42](../results/not_improved/AMT_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/not_improved/AMT_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/not_improved/AMT_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | MT-TC | 1.95 / 1.74 / 1.81 | [42](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | LT_DTH | 1.92 / 1.93 / 1.90 | [42](../results/not_improved/LT_DTH_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/not_improved/LT_DTH_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/not_improved/LT_DTH_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | SAMT | 1.75 / 1.75 / 1.78 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| ECG | Reference | 38.64 / 33.38 / 29.60 | [42](../results/valid/reference/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/reference/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/reference/ECG/seed_456/COST_RESULTS.json) |
| ECG | DTH | 43.81 / 43.67 / 44.26 | [42](../results/valid/DTH_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | MT | 45.26 / 44.52 / 44.23 | [42](../results/valid/MT_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | AMT | 46.88 / 46.20 / 44.80 | [42](../results/not_improved/AMT_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/not_improved/AMT_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/not_improved/AMT_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | MT-TC | 45.78 / 46.50 / 46.73 | [42](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | LT_DTH | 46.02 / 46.99 / 44.72 | [42](../results/not_improved/LT_DTH_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/not_improved/LT_DTH_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/not_improved/LT_DTH_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | SAMT | 42.77 / 43.33 / 41.18 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |

Seed values are in order 42 / 123 / 456. Links retain checkpoint identities, configs, measurements and environment evidence.

Accuracy sources: [paper](paper-results.md) and the ordinary-inference test records below. Values were checked against saved final-test metrics and cost-record reproduced accuracies. Canonical reference SHD* records also agree with `REFERENCE_RESULT.json`; reference SSC seed 42 uses `TEST_RESULTS.json` because its retained training CSV has no final-test row. No inference or training was rerun. Archived speaker-holdout and exploratory SA-MT fresh-tuning trials are excluded.

| Dataset | Model | Test accuracy (%) by seed 42 / 123 / 456 | Source metrics by seed |
|---|---|---|---|
| SHD* | Reference | 92.93 / 92.89 / 93.64 | [42](../results/valid/reference/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/reference/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/reference/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | DTH | 94.70 / 94.88 / 93.90 | [42](../results/valid/DTH_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/DTH_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/DTH_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | MT | 94.88 / 94.52 / 94.43 | [42](../results/valid/MT_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/MT_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/MT_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | AMT | 94.48 / 90.64 / 93.02 | [42](../results/not_improved/AMT_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/AMT_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/AMT_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | MT-TC | 94.74 / 92.84 / 93.82 | [42](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | LT_DTH | 94.66 / 93.42 / 93.86 | [42](../results/not_improved/LT_DTH_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/LT_DTH_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/LT_DTH_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | SAMT | 94.17 / 93.15 / 93.86 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | Reference | 80.69 / 80.33 / 80.38 | [42](../results/valid/reference/SSC/seed_42/TEST_RESULTS.json), [123](../results/valid/reference/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/reference/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | DTH | 80.46 / 79.84 / 79.56 | [42](../results/valid/DTH_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/DTH_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/DTH_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | MT | 80.01 / 79.98 / 80.23 | [42](../results/valid/MT_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/MT_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/MT_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | AMT | 80.24 / 80.45 / 80.08 | [42](../results/not_improved/AMT_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/AMT_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/AMT_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | MT-TC | 80.12 / 80.05 / 80.26 | [42](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | LT_DTH | 79.95 / 80.25 / 80.31 | [42](../results/not_improved/LT_DTH_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/LT_DTH_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/LT_DTH_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | SAMT | 80.13 / 80.32 / 79.84 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | Reference | 88.13 / 88.72 / 88.11 | [42](../results/valid/reference/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/reference/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/reference/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | DTH | 87.81 / 87.81 / 87.27 | [42](../results/valid/DTH_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/DTH_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/DTH_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | MT | 87.26 / 88.19 / 87.98 | [42](../results/valid/MT_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/MT_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/MT_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | AMT | 87.65 / 87.82 / 88.02 | [42](../results/not_improved/AMT_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/AMT_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/AMT_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | MT-TC | 87.78 / 87.98 / 88.62 | [42](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | LT_DTH | 88.33 / 87.11 / 87.58 | [42](../results/not_improved/LT_DTH_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/LT_DTH_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/LT_DTH_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | SAMT | 88.26 / 88.65 / 88.27 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |

[SSC transfer verification](../results/provenance/ssc-local-import-2026-10-21.json).
