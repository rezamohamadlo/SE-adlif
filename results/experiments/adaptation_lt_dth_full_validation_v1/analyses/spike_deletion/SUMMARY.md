# SHD* full-validation analysis: spike_deletion

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 1631; paired records across all groups: 19572.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 1631 | 0.061183 | 0.055481 | +0.005702 | 99.08% | 98.96% | -0.12 pp |
| 123 | delete_0.1 | 0 | 1631 | 0.069144 | 0.070276 | -0.001132 | 98.47% | 98.22% | -0.25 pp |
| 123 | delete_0.1 | 1 | 1631 | 0.068481 | 0.077625 | -0.009144 | 98.71% | 98.04% | -0.67 pp |
| 123 | delete_0.1 | 2 | 1631 | 0.065031 | 0.069082 | -0.004050 | 98.47% | 98.59% | +0.12 pp |
| 42 | clean | 0 | 1631 | 0.052656 | 0.058605 | -0.005950 | 99.14% | 98.96% | -0.18 pp |
| 42 | delete_0.1 | 0 | 1631 | 0.067238 | 0.067431 | -0.000193 | 98.47% | 98.84% | +0.37 pp |
| 42 | delete_0.1 | 1 | 1631 | 0.060053 | 0.064936 | -0.004884 | 98.90% | 98.53% | -0.37 pp |
| 42 | delete_0.1 | 2 | 1631 | 0.060328 | 0.064482 | -0.004155 | 98.59% | 98.59% | +0.00 pp |
| 456 | clean | 0 | 1631 | 0.082774 | 0.057639 | +0.025135 | 99.02% | 99.51% | +0.49 pp |
| 456 | delete_0.1 | 0 | 1631 | 0.086758 | 0.066597 | +0.020160 | 98.65% | 99.08% | +0.43 pp |
| 456 | delete_0.1 | 1 | 1631 | 0.083414 | 0.075714 | +0.007701 | 98.41% | 98.77% | +0.37 pp |
| 456 | delete_0.1 | 2 | 1631 | 0.085467 | 0.067104 | +0.018363 | 98.34% | 98.96% | +0.61 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.065538 ± 0.015524 | 0.057242 ± 0.001600 | +0.008296 ± 0.015704 | 99.08 ± 0.06 | 99.14 ± 0.32 | +0.06 ± 0.37 |
| delete_0.1 | 0 | 3 | complete | 0.074380 ± 0.010762 | 0.068102 ± 0.001929 | +0.006278 ± 0.012031 | 98.53 ± 0.11 | 98.71 ± 0.44 | +0.18 ± 0.37 |
| delete_0.1 | 1 | 3 | complete | 0.070649 ± 0.011831 | 0.072758 ± 0.006841 | -0.002109 ± 0.008758 | 98.67 ± 0.25 | 98.45 ± 0.37 | -0.22 ± 0.54 |
| delete_0.1 | 2 | 3 | complete | 0.070275 ± 0.013365 | 0.066889 ± 0.002307 | +0.003386 ± 0.012970 | 98.47 ± 0.12 | 98.71 ± 0.21 | +0.25 ± 0.32 |

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
| delete_0.1 | 3 | -0.00577750 +/- 0.00750052 |
