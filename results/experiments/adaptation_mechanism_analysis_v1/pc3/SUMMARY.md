# SHD* paired validation pilot

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 256; paired records across all groups: 3072.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 256 | 0.050395 | 0.003242 | +0.047153 | 99.61% | 100.00% | +0.39 pp |
| 123 | delete_0.1 | 0 | 256 | 0.044384 | 0.007765 | +0.036619 | 99.61% | 99.61% | +0.00 pp |
| 123 | delete_0.1 | 1 | 256 | 0.050569 | 0.009433 | +0.041136 | 99.61% | 99.61% | +0.00 pp |
| 123 | delete_0.1 | 2 | 256 | 0.046157 | 0.011640 | +0.034516 | 99.61% | 99.61% | +0.00 pp |
| 42 | clean | 0 | 256 | 0.049249 | 0.056841 | -0.007592 | 99.22% | 99.61% | +0.39 pp |
| 42 | delete_0.1 | 0 | 256 | 0.060278 | 0.067055 | -0.006777 | 99.22% | 98.83% | -0.39 pp |
| 42 | delete_0.1 | 1 | 256 | 0.067660 | 0.073462 | -0.005802 | 99.22% | 98.83% | -0.39 pp |
| 42 | delete_0.1 | 2 | 256 | 0.059581 | 0.061507 | -0.001926 | 99.22% | 99.61% | +0.39 pp |
| 456 | clean | 0 | 256 | 0.026927 | 0.055336 | -0.028409 | 99.61% | 98.83% | -0.78 pp |
| 456 | delete_0.1 | 0 | 256 | 0.023965 | 0.038950 | -0.014985 | 99.61% | 99.22% | -0.39 pp |
| 456 | delete_0.1 | 1 | 256 | 0.043512 | 0.062651 | -0.019138 | 99.61% | 99.22% | -0.39 pp |
| 456 | delete_0.1 | 2 | 256 | 0.051564 | 0.044909 | +0.006655 | 99.22% | 99.22% | +0.00 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.042191 ± 0.013231 | 0.038473 ± 0.030520 | +0.003717 ± 0.039030 | 99.48 ± 0.23 | 99.48 ± 0.60 | +0.00 ± 0.68 |
| delete_0.1 | 0 | 3 | complete | 0.042876 ± 0.018203 | 0.037924 ± 0.029658 | +0.004952 ± 0.027729 | 99.48 ± 0.23 | 99.22 ± 0.39 | -0.26 ± 0.23 |
| delete_0.1 | 1 | 3 | complete | 0.053914 ± 0.012416 | 0.048515 ± 0.034275 | +0.005398 ± 0.031660 | 99.48 ± 0.23 | 99.22 ± 0.39 | -0.26 ± 0.23 |
| delete_0.1 | 2 | 3 | complete | 0.052434 ± 0.006754 | 0.039352 ± 0.025394 | +0.013082 ± 0.019052 | 99.35 ± 0.23 | 99.48 ± 0.23 | +0.13 ± 0.23 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 256 | +0.0309 | - |
| 123 | `fast_slow_contrast` | 256 | -0.0754 | - |
| 123 | `autocorr_tau_ms` | 256 | +0.0541 | - |
| 123 | `active_channel_fraction` | 256 | -0.0966 | - |
| 123 | `duration_ms` | 256 | -0.0961 | - |
| 42 | `mean_rate_hz` | 256 | -0.0432 | - |
| 42 | `fast_slow_contrast` | 256 | +0.0350 | - |
| 42 | `autocorr_tau_ms` | 256 | -0.0680 | - |
| 42 | `active_channel_fraction` | 256 | +0.0162 | - |
| 42 | `duration_ms` | 256 | +0.0539 | - |
| 456 | `mean_rate_hz` | 256 | +0.0452 | - |
| 456 | `fast_slow_contrast` | 256 | +0.0661 | - |
| 456 | `autocorr_tau_ms` | 256 | +0.0060 | - |
| 456 | `active_channel_fraction` | 256 | +0.1243 | - |
| 456 | `duration_ms` | 256 | +0.0335 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.
