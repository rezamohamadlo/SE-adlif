# SSC paired full validation: SA_MT_SE_adLIF versus MT_SE_adLIF / time_scaling

**Full official SSC validation: 9981 records per condition.** Clean results also supply the saved compatibility anchors. These are exploratory validation diagnostics, with the original class frequencies. They are not independent confirmation or final-test results.

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 9981; paired records across all groups: 89829.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 9981 | 0.640417 | 0.632473 | +0.007944 | 82.00% | 81.95% | -0.05 pp |
| 123 | time_scale_0.9 | 0 | 9981 | 0.704700 | 0.712167 | -0.007467 | 80.44% | 79.99% | -0.45 pp |
| 123 | time_scale_1.1 | 0 | 9981 | 0.679292 | 0.676015 | +0.003277 | 80.33% | 80.39% | +0.06 pp |
| 42 | clean | 0 | 9981 | 0.612064 | 0.613298 | -0.001234 | 82.00% | 81.82% | -0.18 pp |
| 42 | time_scale_0.9 | 0 | 9981 | 0.678149 | 0.667308 | +0.010841 | 80.32% | 79.98% | -0.34 pp |
| 42 | time_scale_1.1 | 0 | 9981 | 0.665561 | 0.681482 | -0.015921 | 80.02% | 79.13% | -0.89 pp |
| 456 | clean | 0 | 9981 | 0.612423 | 0.636652 | -0.024229 | 82.47% | 81.89% | -0.58 pp |
| 456 | time_scale_0.9 | 0 | 9981 | 0.696732 | 0.724687 | -0.027955 | 80.09% | 79.60% | -0.49 pp |
| 456 | time_scale_1.1 | 0 | 9981 | 0.645333 | 0.677343 | -0.032010 | 80.79% | 80.09% | -0.70 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.621635 ± 0.016267 | 0.627474 ± 0.012454 | -0.005840 ± 0.016573 | 82.15 ± 0.27 | 81.88 ± 0.07 | -0.27 ± 0.28 |
| time_scale_0.9 | 0 | 3 | complete | 0.693193 ± 0.013625 | 0.701387 ± 0.030170 | -0.008194 ± 0.019408 | 80.29 ± 0.18 | 79.86 ± 0.22 | -0.43 ± 0.08 |
| time_scale_1.1 | 0 | 3 | complete | 0.663395 ± 0.017083 | 0.678280 ± 0.002852 | -0.014885 ± 0.017666 | 80.38 ± 0.39 | 79.87 ± 0.66 | -0.51 ± 0.50 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 9981 | -0.0244 | - |
| 123 | `fast_slow_contrast` | 9981 | +0.0187 | - |
| 123 | `autocorr_tau_ms` | 9979 | -0.0162 | - |
| 123 | `active_channel_fraction` | 9981 | -0.0154 | - |
| 123 | `duration_ms` | 9981 | +0.0043 | - |
| 42 | `mean_rate_hz` | 9981 | +0.0215 | - |
| 42 | `fast_slow_contrast` | 9981 | -0.0167 | - |
| 42 | `autocorr_tau_ms` | 9979 | +0.0158 | - |
| 42 | `active_channel_fraction` | 9981 | -0.0135 | - |
| 42 | `duration_ms` | 9981 | -0.0166 | - |
| 456 | `mean_rate_hz` | 9981 | -0.0334 | - |
| 456 | `fast_slow_contrast` | 9981 | +0.0127 | - |
| 456 | `autocorr_tau_ms` | 9979 | +0.0084 | - |
| 456 | `active_channel_fraction` | 9981 | +0.0042 | - |
| 456 | `duration_ms` | 9981 | +0.0007 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors SA_MT_SE_adLIF relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
| time_scale_0.9 | 3 | -0.00235436 +/- 0.01379424 |
| time_scale_1.1 | 3 | -0.00904490 +/- 0.00512817 |
