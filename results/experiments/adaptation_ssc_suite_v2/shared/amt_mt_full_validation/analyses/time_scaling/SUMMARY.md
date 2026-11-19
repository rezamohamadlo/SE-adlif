# SSC paired full validation: time_scaling

**Full official SSC validation: 9981 records per condition.** Clean results also supply the saved compatibility anchors. These are exploratory validation diagnostics, with the original class frequencies. They are not independent confirmation or final-test results.

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 9981; paired records across all groups: 89829.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 9981 | 0.640417 | 0.624710 | +0.015707 | 82.00% | 81.95% | -0.05 pp |
| 123 | time_scale_0.9 | 0 | 9981 | 0.704700 | 0.682917 | +0.021783 | 80.44% | 80.57% | +0.13 pp |
| 123 | time_scale_1.1 | 0 | 9981 | 0.679292 | 0.663957 | +0.015334 | 80.33% | 80.60% | +0.27 pp |
| 42 | clean | 0 | 9981 | 0.612064 | 0.646666 | -0.034603 | 82.00% | 82.25% | +0.25 pp |
| 42 | time_scale_0.9 | 0 | 9981 | 0.678149 | 0.715843 | -0.037694 | 80.32% | 80.66% | +0.34 pp |
| 42 | time_scale_1.1 | 0 | 9981 | 0.665561 | 0.688107 | -0.022546 | 80.02% | 80.22% | +0.20 pp |
| 456 | clean | 0 | 9981 | 0.612423 | 0.604784 | +0.007639 | 82.47% | 81.95% | -0.52 pp |
| 456 | time_scale_0.9 | 0 | 9981 | 0.696732 | 0.669986 | +0.026746 | 80.09% | 80.03% | -0.06 pp |
| 456 | time_scale_1.1 | 0 | 9981 | 0.645333 | 0.666998 | -0.021666 | 80.79% | 79.92% | -0.87 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.621635 ± 0.016267 | 0.625387 ± 0.020949 | -0.003752 ± 0.027020 | 82.15 ± 0.27 | 82.05 ± 0.17 | -0.11 ± 0.39 |
| time_scale_0.9 | 0 | 3 | complete | 0.693193 ± 0.013625 | 0.689582 ± 0.023644 | +0.003611 ± 0.035858 | 80.29 ± 0.18 | 80.42 ± 0.34 | +0.14 ± 0.20 |
| time_scale_1.1 | 0 | 3 | complete | 0.663395 ± 0.017083 | 0.673021 ± 0.013153 | -0.009626 ± 0.021620 | 80.38 ± 0.39 | 80.25 ± 0.34 | -0.13 ± 0.64 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 9981 | -0.0100 | - |
| 123 | `fast_slow_contrast` | 9981 | +0.0120 | - |
| 123 | `autocorr_tau_ms` | 9979 | -0.0148 | - |
| 123 | `active_channel_fraction` | 9981 | -0.0014 | - |
| 123 | `duration_ms` | 9981 | -0.0180 | - |
| 42 | `mean_rate_hz` | 9981 | -0.0015 | - |
| 42 | `fast_slow_contrast` | 9981 | -0.0027 | - |
| 42 | `autocorr_tau_ms` | 9979 | -0.0238 | - |
| 42 | `active_channel_fraction` | 9981 | -0.0121 | - |
| 42 | `duration_ms` | 9981 | -0.0235 | - |
| 456 | `mean_rate_hz` | 9981 | -0.0315 | - |
| 456 | `fast_slow_contrast` | 9981 | +0.0227 | - |
| 456 | `autocorr_tau_ms` | 9979 | -0.0139 | - |
| 456 | `active_channel_fraction` | 9981 | -0.0105 | - |
| 456 | `duration_ms` | 9981 | -0.0256 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors AMT relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
| time_scale_0.9 | 3 | +0.00736368 +/- 0.01115541 |
| time_scale_1.1 | 3 | -0.00587333 +/- 0.02122214 |
