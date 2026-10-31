# Paper and proposed-model comparison

## Test accuracy

### Without TTA

| Dataset | Paper | DTH (Δ paper) | MT (Δ paper) | AMT (Δ paper) | SA-MT (Δ paper) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | 93.79 ± 0.76 | **94.49 ± 0.52** (+0.70 pp) | **94.61 ± 0.23** (+0.82 pp) | 92.71 ± 1.94 (-1.08 pp) | 93.73 ± 0.52 (-0.06 pp) |
| SSC | 80.44 ± 0.26 | 79.95 ± 0.46 (-0.49 pp) | 80.07 ± 0.14 (-0.37 pp) | 80.26 ± 0.18 (-0.18 pp) | 80.10 ± 0.25 (-0.34 pp) |
| ECG | 88.18 ± 0.36 | 87.63 ± 0.31 (-0.55 pp) | 87.81 ± 0.49 (-0.37 pp) | 87.83 ± 0.18 (-0.35 pp) | **88.39 ± 0.22** (+0.21 pp) |

Snapshot: 29 October 2026. Models: DTH, MT, AMT and SA-MT.

Main comparisons use **ordinary inference without test-time augmentation (TTA)** for accuracy and cost. Accuracy (%) is mean ± sample SD across matching training seeds 42, 123 and 456 (n=3), using final test evaluation of validation-selected checkpoints. **Bold** marks a local mean above the paper mean. Δ paper = local mean minus the published paper mean, in percentage points (pp), calculated before rounding. Paper sample sizes are SHD*/ECG n=20 and SSC n=5. These descriptive comparisons do not establish statistical significance. Exploratory TTA accuracy is presented separately below and does not determine the main reported result.

All four models share the dataset-specific baseline optimization, preprocessing and budget. SSC uses `ssc_mean_loss_v1`. SHD* uses a fixed 20% training-derived validation split with split seed 42.

### With TTA (exploratory)

| Dataset | Paper | DTH (Δ paper) | MT (Δ paper) | AMT (Δ paper) | SA-MT (Δ paper) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | 93.79 ± 0.76 | **94.64 ± 0.54** (+0.85 pp) | **94.61 ± 0.31** (+0.82 pp) | 92.76 ± 2.11 (-1.03 pp) | 93.77 ± 0.61 (-0.02 pp) |
| SSC | 80.44 ± 0.26 | 80.34 ± 0.49 (-0.10 pp) | 80.39 ± 0.01 (-0.05 pp) | **80.72 ± 0.28** (+0.28 pp) | **80.54 ± 0.22** (+0.10 pp) |
| ECG | 88.18 ± 0.36 | **88.40 ± 0.17** (+0.22 pp) | **88.53 ± 0.54** (+0.35 pp) | **88.60 ± 0.17** (+0.42 pp) | **89.06 ± 0.28** (+0.88 pp) |

TTA uses frozen validation-selected checkpoints and averages class probabilities over input gains 1.0, 0.95 and 1.05 (three forward passes). Values are mean ± sample SD across training seeds 42, 123 and 456 (n=3); Δ paper uses the unrounded TTA mean. All entries in this table use TTA, including when ordinary inference has a higher score.

These evaluations are exploratory because the test results were previously inspected; they do not establish independent improvements or statistical significance. The cost tables below describe ordinary inference and must not be interpreted as TTA latency or energy.

Sources: [DTH SHD*/ECG](../results/experiments/dth_tta_v1/RESULTS.md), [DTH SSC](../results/experiments/dth_tta_v2/RESULTS.md), [MT SHD*/ECG](../results/experiments/mt_tta_v1/RESULTS.md), [MT SSC](../results/experiments/mt_tta_v2/RESULTS.md), [AMT](../results/experiments/amt_tta_v1/RESULTS.md), [SA-MT](../results/experiments/sa_mt_tta_v1/RESULTS.md).

## Inference time

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 1.61 ± 0.04 | 1.67 ± 0.03 | 1.84 ± 0.12 | 1.59 ± 0.01 |
| SSC | Not reported | 1.74 ± 0.16 | 1.70 ± 0.14 | 2.02 ± 0.11 | 1.76 ± 0.02 |
| ECG | Not reported | 43.91 ± 0.31 | 44.67 ± 0.53 | 45.96 ± 1.06 | 42.43 ± 1.12 |

All cost tables use ordinary inference without TTA. Timing is forward-only milliseconds/sample, mean ± sample SD across three seeds.

Saved timing uses three repeats, two warm-up batches, batch sizes 256 for SHD*/SSC and 64 for ECG, RTX 4090 and high matmul precision. It excludes loading, transfer, loss and temporal prediction aggregation. Sessions and evaluator environments differ; these are descriptive measurements, not a controlled hardware speed benchmark.

## Parameters, activity and computational cost

### Total parameters

| Dataset | Paper | DTH (Δ paper) | MT (Δ paper) | AMT (Δ paper) | SA-MT (Δ paper) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | 0.45M | 451,484 (≈+1,484 params) | 452,200 (≈+2,200 params) | 452,920 (≈+2,920 params) | 452,920 (≈+2,920 params) |
| SSC | 1.6M | 1,691,354 (≈+91,354 params) | 1,692,790 (≈+92,790 params) | 1,694,230 (≈+94,230 params) | 1,694,230 (≈+94,230 params) |
| ECG | 4.6k | 4,768 (≈+168 params) | 4,836 (≈+236 params) | 4,908 (≈+308 params) | 4,908 (≈+308 params) |

Published counts are rounded: 0.45M = 450,000, 1.6M = 1,600,000 and 4.6k = 4,600. Count differences from those rounded baselines are approximate (≈), in parameters.

All statistics below come from the same per-seed cost records. Parameter counts are exact and identical across seeds. Other values are mean ± sample SD (n=3); zero SD means equal counted costs, not measurement certainty.

### Trainable parameters

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 450,744 | 451,460 | 452,180 | 452,180 |
| SSC | Not reported | 1,689,879 | 1,691,315 | 1,692,755 | 1,692,755 |
| ECG | Not reported | 4,690 | 4,758 | 4,830 | 4,830 |



### Hidden spikes per sample

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 10,513.9 ± 612.6 | 10,637.4 ± 608.6 | 10,739.6 ± 375.6 | 10,712.1 ± 587.9 |
| SSC | Not reported | 45,601.5 ± 4,030.9 | 49,500.0 ± 580.7 | 49,383.4 ± 2,491.1 | 49,151.0 ± 1,644.4 |
| ECG | Not reported | 30,057.7 ± 2,822.3 | 26,665.3 ± 378.0 | 27,536.2 ± 386.6 | 27,166.5 ± 950.0 |



### Event-driven synaptic-operation estimate (million SynOps/sample)

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 9.78 ± 0.30 | 9.86 ± 0.42 | 9.98 ± 0.12 | 9.87 ± 0.37 |
| SSC | Not reported | 53.51 ± 4.73 | 58.35 ± 0.32 | 58.29 ± 3.00 | 58.17 ± 2.31 |
| ECG | Not reported | 1.61 ± 0.17 | 1.42 ± 0.04 | 1.48 ± 0.03 | 1.45 ± 0.05 |



### Hidden spike sparsity (%)

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 90.76 ± 0.26 / 97.56 ± 0.48 | 90.67 ± 0.62 / 97.52 ± 0.10 | 90.39 ± 0.11 / 97.68 ± 0.50 | 90.70 ± 0.47 / 97.40 ± 0.20 |
| SSC | Not reported | 91.18 ± 1.14 / 87.71 ± 0.72 | 89.89 ± 0.12 / 87.20 ± 0.37 | 89.87 ± 0.77 / 87.27 ± 0.40 | 89.83 ± 0.73 / 87.42 ± 0.16 |
| ECG | Not reported | 75.17 ± 3.99 / 60.66 ± 2.86 | 78.77 ± 2.62 / 64.30 ± 2.49 | 77.25 ± 1.12 / 63.96 ± 0.78 | 77.69 ± 1.22 / 64.31 ± 2.18 |

Each cell reports L1 / L2. Sparsity is 100 minus the saved firing percentage over stored sample timesteps, including minimum padding and burn-in but excluding added batch padding.

### Dense matrix cost (GFLOPs/sample)

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 | 0.242895 ± 0.000000 |
| SSC | Not reported | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 | 1.008720 ± 0.000000 |
| ECG | Not reported | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 | 0.011053 ± 0.000000 |



### Matrix plus scalar arithmetic cost (GFLOPs/sample)

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 0.250372 ± 0.000000 | 0.249006 ± 0.000000 | 0.250769 ± 0.000000 | 0.249789 ± 0.000000 |
| SSC | Not reported | 1.025202 ± 0.000000 | 1.022190 ± 0.000000 | 1.026077 ± 0.000000 | 1.023918 ± 0.000000 |
| ECG | Not reported | 0.014678 ± 0.000000 | 0.014033 ± 0.000000 | 0.014879 ± 0.000000 | 0.014407 ± 0.000000 |



### Logical operation estimate (GOp/sample)

| Dataset | Paper | DTH | MT | AMT | SA-MT |
| --- | ---: | ---: | ---: | ---: | ---: |
| SHD* | Not reported | 0.251353 ± 0.000000 | 0.249597 ± 0.000000 | 0.251555 ± 0.000000 | 0.250380 ± 0.000000 |
| SSC | Not reported | 1.027365 ± 0.000000 | 1.023493 ± 0.000000 | 1.027811 ± 0.000000 | 1.025221 ± 0.000000 |
| ECG | Not reported | 0.015150 ± 0.000000 | 0.014322 ± 0.000000 | 0.015260 ± 0.000000 | 0.014696 ± 0.000000 |

Hidden spikes sum both hidden layers and exclude output activity. SynOps estimate ideal event-driven fan-out additions; they do not measure dense GPU operations or energy. ECG analog-input projection is excluded from SynOps and counted separately as dense MACs in source records.

Dense matrix cost counts each multiply-add as two FLOPs. Arithmetic adds counted neuron scalar arithmetic, excluding nonlinearities and comparisons. Logical GOp additionally includes counted nonlinear/comparison operations and is not hardware GFLOPs. Compute uses executed batch-padded sequence extent and excludes memory traffic, loading, transfer, loss and temporal aggregation. Paper classification activity/compute values are unavailable; rounded parameter counts come from Table 2 (PDF page 11), which does not separate trainable counts. Audio-bottleneck spike rates are not comparable classification activity.

## Per-seed measurements and sources

| Dataset | Model | Per-seed ms/sample | Cost records by seed |
|---|---:|---:|---:|
| SHD* | DTH | 1.66 / 1.60 / 1.58 | [42](../results/valid/DTH_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | MT | 1.64 / 1.68 / 1.68 | [42](../results/valid/MT_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | AMT | 1.98 / 1.77 / 1.78 | [42](../results/not_improved/AMT_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/not_improved/AMT_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/not_improved/AMT_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SHD* | SA-MT | 1.59 / 1.58 / 1.59 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_42/COST_RESULTS.json), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_123/COST_RESULTS.json), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_456/COST_RESULTS.json) |
| SSC | DTH | 1.61 / 1.92 / 1.70 | [42](../results/valid/DTH_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | MT | 1.56 / 1.85 / 1.69 | [42](../results/valid/MT_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | AMT | 1.98 / 1.94 / 2.15 | [42](../results/not_improved/AMT_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/not_improved/AMT_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/not_improved/AMT_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| SSC | SA-MT | 1.75 / 1.75 / 1.78 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_42/COST_RESULTS.json), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_123/COST_RESULTS.json), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_456/COST_RESULTS.json) |
| ECG | DTH | 43.81 / 43.67 / 44.26 | [42](../results/valid/DTH_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/DTH_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/DTH_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | MT | 45.26 / 44.52 / 44.23 | [42](../results/valid/MT_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/valid/MT_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/valid/MT_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | AMT | 46.88 / 46.20 / 44.80 | [42](../results/not_improved/AMT_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/not_improved/AMT_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/not_improved/AMT_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |
| ECG | SA-MT | 42.77 / 43.33 / 41.18 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_42/COST_RESULTS.json), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_123/COST_RESULTS.json), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_456/COST_RESULTS.json) |

Seed values are in order 42 / 123 / 456. Links retain checkpoint identities, configs, measurements and environment evidence.

Accuracy sources: [paper](paper-results.md) and the ordinary-inference test records below. Values were checked against each seed's `metrics.csv`; no inference or training was rerun.

| Dataset | Model | Test accuracy (%) by seed 42 / 123 / 456 | Source metrics by seed |
|---|---|---|---|
| SHD* | DTH | 94.70 / 94.88 / 93.90 | [42](../results/valid/DTH_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/DTH_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/DTH_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | MT | 94.88 / 94.52 / 94.43 | [42](../results/valid/MT_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/MT_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/MT_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | AMT | 94.48 / 90.64 / 93.02 | [42](../results/not_improved/AMT_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/AMT_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/AMT_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SHD* | SA-MT | 94.17 / 93.15 / 93.86 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SHD/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | DTH | 80.46 / 79.84 / 79.56 | [42](../results/valid/DTH_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/DTH_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/DTH_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | MT | 80.01 / 79.98 / 80.23 | [42](../results/valid/MT_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/MT_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/MT_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | AMT | 80.24 / 80.45 / 80.08 | [42](../results/not_improved/AMT_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/AMT_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/AMT_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| SSC | SA-MT | 80.13 / 80.32 / 79.84 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/SSC/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | DTH | 87.81 / 87.81 / 87.27 | [42](../results/valid/DTH_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/DTH_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/DTH_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | MT | 87.26 / 88.19 / 87.98 | [42](../results/valid/MT_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/valid/MT_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/valid/MT_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | AMT | 87.65 / 87.82 / 88.02 | [42](../results/not_improved/AMT_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/not_improved/AMT_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/not_improved/AMT_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |
| ECG | SA-MT | 88.26 / 88.65 / 88.27 | [42](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_42/logs/mlp_snn/version_0/metrics.csv), [123](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_123/logs/mlp_snn/version_0/metrics.csv), [456](../results/experiments/sa_mt_v1/SA_MT_SE_adLIF/ECG/seed_456/logs/mlp_snn/version_0/metrics.csv) |

[SSC transfer verification](../results/provenance/ssc-local-import-2026-10-21.json).
