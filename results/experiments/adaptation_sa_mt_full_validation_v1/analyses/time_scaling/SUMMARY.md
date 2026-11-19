# SHD* full-validation analysis: time_scaling

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 1631; paired records across all groups: 14679.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 1631 | 0.057002 | 0.069464 | -0.012461 | 99.20% | 99.14% | -0.06 pp |
| 123 | time_scale_0.9 | 0 | 1631 | 0.071126 | 0.105715 | -0.034589 | 98.41% | 98.10% | -0.31 pp |
| 123 | time_scale_1.1 | 0 | 1631 | 0.080842 | 0.091025 | -0.010183 | 98.59% | 98.53% | -0.06 pp |
| 42 | clean | 0 | 1631 | 0.093871 | 0.096086 | -0.002216 | 99.26% | 98.90% | -0.37 pp |
| 42 | time_scale_0.9 | 0 | 1631 | 0.130714 | 0.148151 | -0.017437 | 98.04% | 97.61% | -0.43 pp |
| 42 | time_scale_1.1 | 0 | 1631 | 0.111157 | 0.128509 | -0.017352 | 98.28% | 98.04% | -0.25 pp |
| 456 | clean | 0 | 1631 | 0.048669 | 0.046891 | +0.001777 | 99.33% | 99.39% | +0.06 pp |
| 456 | time_scale_0.9 | 0 | 1631 | 0.075418 | 0.072081 | +0.003336 | 98.22% | 98.22% | +0.00 pp |
| 456 | time_scale_1.1 | 0 | 1631 | 0.083111 | 0.066972 | +0.016138 | 98.41% | 98.84% | +0.43 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.066514 ± 0.024055 | 0.070814 ± 0.024625 | -0.004300 ± 0.007345 | 99.26 ± 0.06 | 99.14 ± 0.25 | -0.12 ± 0.22 |
| time_scale_0.9 | 0 | 3 | complete | 0.092419 ± 0.033234 | 0.108649 ± 0.038119 | -0.016230 ± 0.018991 | 98.22 ± 0.18 | 97.98 ± 0.32 | -0.25 ± 0.22 |
| time_scale_1.1 | 0 | 3 | complete | 0.091703 ± 0.016885 | 0.095502 ± 0.031012 | -0.003799 ± 0.017634 | 98.43 ± 0.15 | 98.47 ± 0.40 | +0.04 ± 0.35 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 1631 | +0.0029 | - |
| 123 | `fast_slow_contrast` | 1631 | -0.0408 | - |
| 123 | `autocorr_tau_ms` | 1631 | -0.0118 | - |
| 123 | `active_channel_fraction` | 1631 | -0.0561 | - |
| 123 | `duration_ms` | 1631 | -0.0662 | - |
| 42 | `mean_rate_hz` | 1631 | -0.0243 | - |
| 42 | `fast_slow_contrast` | 1631 | -0.0010 | - |
| 42 | `autocorr_tau_ms` | 1631 | +0.0329 | - |
| 42 | `active_channel_fraction` | 1631 | +0.0053 | - |
| 42 | `duration_ms` | 1631 | +0.0148 | - |
| 456 | `mean_rate_hz` | 1631 | +0.0294 | - |
| 456 | `fast_slow_contrast` | 1631 | +0.0196 | - |
| 456 | `autocorr_tau_ms` | 1631 | -0.0341 | - |
| 456 | `active_channel_fraction` | 1631 | +0.0125 | - |
| 456 | `duration_ms` | 1631 | +0.0117 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors SA-MT relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
| time_scale_0.9 | 3 | -0.01192988 +/- 0.01218133 |
| time_scale_1.1 | 3 | +0.00050080 +/- 0.01482889 |
