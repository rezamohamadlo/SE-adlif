# SHD* full-validation analysis: spike_deletion

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 1631; paired records across all groups: 19572.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 1631 | 0.057002 | 0.064885 | -0.007882 | 99.20% | 99.20% | +0.00 pp |
| 123 | delete_0.1 | 0 | 1631 | 0.064811 | 0.072727 | -0.007916 | 98.90% | 98.90% | +0.00 pp |
| 123 | delete_0.1 | 1 | 1631 | 0.062942 | 0.071016 | -0.008074 | 98.84% | 98.96% | +0.12 pp |
| 123 | delete_0.1 | 2 | 1631 | 0.064266 | 0.074532 | -0.010266 | 98.90% | 98.96% | +0.06 pp |
| 42 | clean | 0 | 1631 | 0.093871 | 0.081114 | +0.012757 | 99.26% | 99.08% | -0.18 pp |
| 42 | delete_0.1 | 0 | 1631 | 0.096825 | 0.086223 | +0.010602 | 98.71% | 98.65% | -0.06 pp |
| 42 | delete_0.1 | 1 | 1631 | 0.104495 | 0.077861 | +0.026634 | 98.71% | 98.84% | +0.12 pp |
| 42 | delete_0.1 | 2 | 1631 | 0.105667 | 0.076205 | +0.029462 | 98.84% | 98.77% | -0.06 pp |
| 456 | clean | 0 | 1631 | 0.048669 | 0.059769 | -0.011101 | 99.33% | 99.26% | -0.06 pp |
| 456 | delete_0.1 | 0 | 1631 | 0.065889 | 0.057644 | +0.008245 | 98.71% | 98.77% | +0.06 pp |
| 456 | delete_0.1 | 1 | 1631 | 0.068053 | 0.063653 | +0.004400 | 98.71% | 98.96% | +0.25 pp |
| 456 | delete_0.1 | 2 | 1631 | 0.068435 | 0.066197 | +0.002239 | 98.77% | 98.84% | +0.06 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.066514 ± 0.024055 | 0.068589 ± 0.011144 | -0.002075 ± 0.012946 | 99.26 ± 0.06 | 99.18 ± 0.09 | -0.08 ± 0.09 |
| delete_0.1 | 0 | 3 | complete | 0.075842 ± 0.018180 | 0.072198 ± 0.014297 | +0.003643 ± 0.010080 | 98.77 ± 0.11 | 98.77 ± 0.12 | +0.00 ± 0.06 |
| delete_0.1 | 1 | 3 | complete | 0.078497 ± 0.022660 | 0.070843 ± 0.007106 | +0.007653 ± 0.017581 | 98.75 ± 0.07 | 98.92 ± 0.07 | +0.16 ± 0.07 |
| delete_0.1 | 2 | 3 | complete | 0.079456 ± 0.022795 | 0.072311 ± 0.005361 | +0.007145 ± 0.020313 | 98.84 ± 0.06 | 98.86 ± 0.09 | +0.02 ± 0.07 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 1631 | -0.0088 | - |
| 123 | `fast_slow_contrast` | 1631 | -0.0200 | - |
| 123 | `autocorr_tau_ms` | 1631 | +0.0110 | - |
| 123 | `active_channel_fraction` | 1631 | -0.0330 | - |
| 123 | `duration_ms` | 1631 | -0.0346 | - |
| 42 | `mean_rate_hz` | 1631 | -0.0340 | - |
| 42 | `fast_slow_contrast` | 1631 | +0.0438 | - |
| 42 | `autocorr_tau_ms` | 1631 | +0.0389 | - |
| 42 | `active_channel_fraction` | 1631 | +0.0415 | - |
| 42 | `duration_ms` | 1631 | +0.0691 | - |
| 456 | `mean_rate_hz` | 1631 | +0.0113 | - |
| 456 | `fast_slow_contrast` | 1631 | -0.0016 | - |
| 456 | `autocorr_tau_ms` | 1631 | -0.0296 | - |
| 456 | `active_channel_fraction` | 1631 | -0.0045 | - |
| 456 | `duration_ms` | 1631 | +0.0001 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors MT-TC relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
| delete_0.1 | 3 | +0.00822264 +/- 0.00853516 |

Training objectives intentionally differ: MT-TC was trained with an auxiliary JS penalty. Reported prediction losses and benefits use supervised cross-entropy only, excluding that penalty. Shared baseline settings do not make these training objectives identical.
