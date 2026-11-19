# SHD* full-validation analysis: spike_deletion

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 1631; paired records across all groups: 19572.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 1631 | 0.057002 | 0.052011 | +0.004991 | 99.20% | 99.08% | -0.12 pp |
| 123 | delete_0.1 | 0 | 1631 | 0.064811 | 0.087544 | -0.022733 | 98.90% | 98.22% | -0.67 pp |
| 123 | delete_0.1 | 1 | 1631 | 0.062942 | 0.084346 | -0.021404 | 98.84% | 97.98% | -0.86 pp |
| 123 | delete_0.1 | 2 | 1631 | 0.064266 | 0.072522 | -0.008256 | 98.90% | 98.10% | -0.80 pp |
| 42 | clean | 0 | 1631 | 0.093871 | 0.072373 | +0.021497 | 99.26% | 98.90% | -0.37 pp |
| 42 | delete_0.1 | 0 | 1631 | 0.096825 | 0.078017 | +0.018808 | 98.71% | 98.47% | -0.25 pp |
| 42 | delete_0.1 | 1 | 1631 | 0.104495 | 0.072644 | +0.031851 | 98.71% | 98.34% | -0.37 pp |
| 42 | delete_0.1 | 2 | 1631 | 0.105667 | 0.075815 | +0.029851 | 98.84% | 98.28% | -0.55 pp |
| 456 | clean | 0 | 1631 | 0.048669 | 0.097188 | -0.048520 | 99.33% | 99.08% | -0.25 pp |
| 456 | delete_0.1 | 0 | 1631 | 0.065889 | 0.091999 | -0.026110 | 98.71% | 98.96% | +0.25 pp |
| 456 | delete_0.1 | 1 | 1631 | 0.068053 | 0.094063 | -0.026011 | 98.71% | 98.84% | +0.12 pp |
| 456 | delete_0.1 | 2 | 1631 | 0.068435 | 0.097239 | -0.028804 | 98.77% | 98.65% | -0.12 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.066514 ± 0.024055 | 0.073857 ± 0.022625 | -0.007344 ± 0.036602 | 99.26 ± 0.06 | 99.02 ± 0.11 | -0.25 ± 0.12 |
| delete_0.1 | 0 | 3 | complete | 0.075842 ± 0.018180 | 0.085853 ± 0.007143 | -0.010012 ± 0.025016 | 98.77 ± 0.11 | 98.55 ± 0.37 | -0.22 ± 0.46 |
| delete_0.1 | 1 | 3 | complete | 0.078497 ± 0.022660 | 0.083685 ± 0.010725 | -0.005188 ± 0.032159 | 98.75 ± 0.07 | 98.39 ± 0.43 | -0.37 ± 0.49 |
| delete_0.1 | 2 | 3 | complete | 0.079456 ± 0.022795 | 0.081859 ± 0.013422 | -0.002403 ± 0.029763 | 98.84 ± 0.06 | 98.34 ± 0.28 | -0.49 ± 0.34 |

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
| delete_0.1 | 3 | +0.00147615 +/- 0.02225297 |
