# SHD* paired validation pilot

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 256; paired records across all groups: 2304.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 256 | 0.050395 | 0.003242 | +0.047153 | 99.61% | 100.00% | +0.39 pp |
| 123 | time_scale_0.9 | 0 | 256 | 0.063651 | 0.018621 | +0.045030 | 99.61% | 99.61% | +0.00 pp |
| 123 | time_scale_1.1 | 0 | 256 | 0.032396 | 0.020687 | +0.011709 | 99.22% | 99.22% | +0.00 pp |
| 42 | clean | 0 | 256 | 0.049249 | 0.056841 | -0.007592 | 99.22% | 99.61% | +0.39 pp |
| 42 | time_scale_0.9 | 0 | 256 | 0.051535 | 0.089328 | -0.037794 | 99.22% | 98.83% | -0.39 pp |
| 42 | time_scale_1.1 | 0 | 256 | 0.052589 | 0.072169 | -0.019580 | 98.83% | 98.83% | +0.00 pp |
| 456 | clean | 0 | 256 | 0.026927 | 0.055336 | -0.028409 | 99.61% | 98.83% | -0.78 pp |
| 456 | time_scale_0.9 | 0 | 256 | 0.030597 | 0.065548 | -0.034951 | 99.22% | 98.83% | -0.39 pp |
| 456 | time_scale_1.1 | 0 | 256 | 0.047190 | 0.057937 | -0.010747 | 99.22% | 98.44% | -0.78 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.042191 ± 0.013231 | 0.038473 ± 0.030520 | +0.003717 ± 0.039030 | 99.48 ± 0.23 | 99.48 ± 0.60 | +0.00 ± 0.68 |
| time_scale_0.9 | 0 | 3 | complete | 0.048594 ± 0.016722 | 0.057832 ± 0.035980 | -0.009238 ± 0.047019 | 99.35 ± 0.23 | 99.09 ± 0.45 | -0.26 ± 0.23 |
| time_scale_1.1 | 0 | 3 | complete | 0.044058 ± 0.010454 | 0.050264 ± 0.026585 | -0.006206 ± 0.016131 | 99.09 ± 0.23 | 98.83 ± 0.39 | -0.26 ± 0.45 |

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
