# SSC paired full validation: LT_DTH_SE_adLIF versus DTH_SE_adLIF / clean

**Full official SSC validation: 9981 records per condition.** Clean results also supply the saved compatibility anchors. These are exploratory validation diagnostics, with the original class frequencies. They are not independent confirmation or final-test results.

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 9981; paired records across all groups: 29943.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 9981 | 0.668702 | 0.626019 | +0.042683 | 81.72% | 81.54% | -0.17 pp |
| 42 | clean | 0 | 9981 | 0.627100 | 0.606780 | +0.020320 | 81.84% | 81.65% | -0.19 pp |
| 456 | clean | 0 | 9981 | 0.627482 | 0.644957 | -0.017475 | 81.18% | 81.59% | +0.40 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.641095 ± 0.023909 | 0.625919 ± 0.019089 | +0.015176 ± 0.030407 | 81.58 ± 0.35 | 81.59 ± 0.05 | +0.01 ± 0.34 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 9981 | +0.0170 | - |
| 123 | `fast_slow_contrast` | 9981 | -0.0163 | - |
| 123 | `autocorr_tau_ms` | 9979 | -0.0195 | - |
| 123 | `active_channel_fraction` | 9981 | -0.0203 | - |
| 123 | `duration_ms` | 9981 | -0.0397 | - |
| 42 | `mean_rate_hz` | 9981 | -0.0089 | - |
| 42 | `fast_slow_contrast` | 9981 | +0.0049 | - |
| 42 | `autocorr_tau_ms` | 9979 | -0.0003 | - |
| 42 | `active_channel_fraction` | 9981 | -0.0084 | - |
| 42 | `duration_ms` | 9981 | -0.0040 | - |
| 456 | `mean_rate_hz` | 9981 | -0.0221 | - |
| 456 | `fast_slow_contrast` | 9981 | +0.0093 | - |
| 456 | `autocorr_tau_ms` | 9979 | +0.0083 | - |
| 456 | `active_channel_fraction` | 9981 | +0.0180 | - |
| 456 | `duration_ms` | 9981 | +0.0206 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors LT_DTH_SE_adLIF relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
