# SHD* full-validation analysis: spike_deletion

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 1631; paired records across all groups: 19572.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 1631 | 0.057002 | 0.069464 | -0.012461 | 99.20% | 99.14% | -0.06 pp |
| 123 | delete_0.1 | 0 | 1631 | 0.064811 | 0.081950 | -0.017140 | 98.90% | 98.65% | -0.25 pp |
| 123 | delete_0.1 | 1 | 1631 | 0.062942 | 0.071859 | -0.008918 | 98.84% | 98.90% | +0.06 pp |
| 123 | delete_0.1 | 2 | 1631 | 0.064266 | 0.078578 | -0.014312 | 98.90% | 98.65% | -0.25 pp |
| 42 | clean | 0 | 1631 | 0.093871 | 0.096086 | -0.002216 | 99.26% | 98.90% | -0.37 pp |
| 42 | delete_0.1 | 0 | 1631 | 0.096825 | 0.109309 | -0.012484 | 98.71% | 98.47% | -0.25 pp |
| 42 | delete_0.1 | 1 | 1631 | 0.104495 | 0.098714 | +0.005781 | 98.71% | 98.41% | -0.31 pp |
| 42 | delete_0.1 | 2 | 1631 | 0.105667 | 0.110970 | -0.005304 | 98.84% | 98.22% | -0.61 pp |
| 456 | clean | 0 | 1631 | 0.048669 | 0.046891 | +0.001777 | 99.33% | 99.39% | +0.06 pp |
| 456 | delete_0.1 | 0 | 1631 | 0.065889 | 0.054691 | +0.011197 | 98.71% | 98.65% | -0.06 pp |
| 456 | delete_0.1 | 1 | 1631 | 0.068053 | 0.058284 | +0.009769 | 98.71% | 98.71% | +0.00 pp |
| 456 | delete_0.1 | 2 | 1631 | 0.068435 | 0.054688 | +0.013747 | 98.77% | 98.84% | +0.06 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.066514 ± 0.024055 | 0.070814 ± 0.024625 | -0.004300 ± 0.007345 | 99.26 ± 0.06 | 99.14 ± 0.25 | -0.12 ± 0.22 |
| delete_0.1 | 0 | 3 | complete | 0.075842 ± 0.018180 | 0.081984 ± 0.027309 | -0.006142 ± 0.015196 | 98.77 ± 0.11 | 98.59 ± 0.11 | -0.18 ± 0.11 |
| delete_0.1 | 1 | 3 | complete | 0.078497 ± 0.022660 | 0.076286 ± 0.020575 | +0.002211 ± 0.009842 | 98.75 ± 0.07 | 98.67 ± 0.25 | -0.08 ± 0.20 |
| delete_0.1 | 2 | 3 | complete | 0.079456 ± 0.022795 | 0.081412 ± 0.028248 | -0.001956 ± 0.014326 | 98.84 ± 0.06 | 98.57 ± 0.31 | -0.27 ± 0.34 |

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
| delete_0.1 | 3 | +0.00233748 +/- 0.00646948 |
