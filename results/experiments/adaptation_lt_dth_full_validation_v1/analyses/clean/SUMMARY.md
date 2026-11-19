# SHD* full-validation analysis: clean

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 1631; paired records across all groups: 4893.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 1631 | 0.061183 | 0.055481 | +0.005702 | 99.08% | 98.96% | -0.12 pp |
| 42 | clean | 0 | 1631 | 0.052656 | 0.058605 | -0.005950 | 99.14% | 98.96% | -0.18 pp |
| 456 | clean | 0 | 1631 | 0.082774 | 0.057639 | +0.025135 | 99.02% | 99.51% | +0.49 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.065538 ± 0.015524 | 0.057242 ± 0.001600 | +0.008296 ± 0.015704 | 99.08 ± 0.06 | 99.14 ± 0.32 | +0.06 ± 0.37 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 1631 | +0.0288 | - |
| 123 | `fast_slow_contrast` | 1631 | -0.0547 | - |
| 123 | `autocorr_tau_ms` | 1631 | -0.0148 | - |
| 123 | `active_channel_fraction` | 1631 | -0.0532 | - |
| 123 | `duration_ms` | 1631 | -0.0619 | - |
| 42 | `mean_rate_hz` | 1631 | -0.0333 | - |
| 42 | `fast_slow_contrast` | 1631 | +0.0523 | - |
| 42 | `autocorr_tau_ms` | 1631 | -0.0265 | - |
| 42 | `active_channel_fraction` | 1631 | +0.0220 | - |
| 42 | `duration_ms` | 1631 | +0.0387 | - |
| 456 | `mean_rate_hz` | 1631 | +0.0327 | - |
| 456 | `fast_slow_contrast` | 1631 | +0.0321 | - |
| 456 | `autocorr_tau_ms` | 1631 | -0.0028 | - |
| 456 | `active_channel_fraction` | 1631 | +0.0339 | - |
| 456 | `duration_ms` | 1631 | +0.0227 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors LT-DTH relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
