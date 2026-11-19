# SHD* full-validation analysis: time_scaling

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 1631; paired records across all groups: 14679.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 1631 | 0.057002 | 0.052011 | +0.004991 | 99.20% | 99.08% | -0.12 pp |
| 123 | time_scale_0.9 | 0 | 1631 | 0.071126 | 0.085336 | -0.014210 | 98.41% | 98.10% | -0.31 pp |
| 123 | time_scale_1.1 | 0 | 1631 | 0.080842 | 0.102213 | -0.021371 | 98.59% | 97.30% | -1.29 pp |
| 42 | clean | 0 | 1631 | 0.093871 | 0.072373 | +0.021497 | 99.26% | 98.90% | -0.37 pp |
| 42 | time_scale_0.9 | 0 | 1631 | 0.130714 | 0.105719 | +0.024995 | 98.04% | 97.79% | -0.25 pp |
| 42 | time_scale_1.1 | 0 | 1631 | 0.111157 | 0.087327 | +0.023830 | 98.28% | 98.34% | +0.06 pp |
| 456 | clean | 0 | 1631 | 0.048669 | 0.097188 | -0.048520 | 99.33% | 99.08% | -0.25 pp |
| 456 | time_scale_0.9 | 0 | 1631 | 0.075418 | 0.134417 | -0.058999 | 98.22% | 98.04% | -0.18 pp |
| 456 | time_scale_1.1 | 0 | 1631 | 0.083111 | 0.101258 | -0.018147 | 98.41% | 98.47% | +0.06 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.066514 ± 0.024055 | 0.073857 ± 0.022625 | -0.007344 ± 0.036602 | 99.26 ± 0.06 | 99.02 ± 0.11 | -0.25 ± 0.12 |
| time_scale_0.9 | 0 | 3 | complete | 0.092419 ± 0.033234 | 0.108491 ± 0.024658 | -0.016072 ± 0.042028 | 98.22 ± 0.18 | 97.98 ± 0.16 | -0.25 ± 0.06 |
| time_scale_1.1 | 0 | 3 | complete | 0.091703 ± 0.016885 | 0.096933 ± 0.008333 | -0.005229 ± 0.025218 | 98.43 ± 0.15 | 98.04 ± 0.64 | -0.39 ± 0.78 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 1631 | -0.0047 | - |
| 123 | `fast_slow_contrast` | 1631 | -0.0006 | - |
| 123 | `autocorr_tau_ms` | 1631 | +0.0167 | - |
| 123 | `active_channel_fraction` | 1631 | -0.0296 | - |
| 123 | `duration_ms` | 1631 | -0.0234 | - |
| 42 | `mean_rate_hz` | 1631 | -0.0233 | - |
| 42 | `fast_slow_contrast` | 1631 | +0.0624 | - |
| 42 | `autocorr_tau_ms` | 1631 | +0.0243 | - |
| 42 | `active_channel_fraction` | 1631 | +0.0529 | - |
| 42 | `duration_ms` | 1631 | +0.0832 | - |
| 456 | `mean_rate_hz` | 1631 | +0.0157 | - |
| 456 | `fast_slow_contrast` | 1631 | -0.0625 | - |
| 456 | `autocorr_tau_ms` | 1631 | +0.0001 | - |
| 456 | `active_channel_fraction` | 1631 | -0.0241 | - |
| 456 | `duration_ms` | 1631 | -0.0452 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors AMT relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
| time_scale_0.9 | 3 | -0.00872799 +/- 0.01145030 |
| time_scale_1.1 | 3 | +0.00211418 +/- 0.02836802 |
