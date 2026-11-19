# SSC paired validation pilot: spike_deletion

**Pilot subset: 256 fixed validation records.** Full clean anchors use 9981 records per checkpoint and are stored separately in `../../VALIDATION_ANCHORS.json`. These tables and feature associations describe the subset only.

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 256; paired records across all groups: 3072.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 256 | 0.888001 | 0.999812 | -0.111811 | 76.17% | 75.00% | -1.17 pp |
| 123 | delete_0.1 | 0 | 256 | 0.906250 | 0.995583 | -0.089332 | 75.00% | 73.05% | -1.95 pp |
| 123 | delete_0.1 | 1 | 256 | 0.831752 | 0.950665 | -0.118912 | 76.95% | 75.78% | -1.17 pp |
| 123 | delete_0.1 | 2 | 256 | 0.895712 | 0.924743 | -0.029031 | 75.00% | 73.05% | -1.95 pp |
| 42 | clean | 0 | 256 | 0.789810 | 0.919410 | -0.129600 | 75.39% | 76.56% | +1.17 pp |
| 42 | delete_0.1 | 0 | 256 | 0.796780 | 0.922565 | -0.125785 | 76.56% | 75.00% | -1.56 pp |
| 42 | delete_0.1 | 1 | 256 | 0.821531 | 0.903502 | -0.081970 | 75.78% | 76.17% | +0.39 pp |
| 42 | delete_0.1 | 2 | 256 | 0.842084 | 0.934943 | -0.092860 | 76.56% | 73.83% | -2.73 pp |
| 456 | clean | 0 | 256 | 0.924454 | 0.877493 | +0.046961 | 74.22% | 76.56% | +2.34 pp |
| 456 | delete_0.1 | 0 | 256 | 0.903823 | 0.806763 | +0.097060 | 74.22% | 76.95% | +2.73 pp |
| 456 | delete_0.1 | 1 | 256 | 0.887018 | 0.830866 | +0.056153 | 74.22% | 78.91% | +4.69 pp |
| 456 | delete_0.1 | 2 | 256 | 0.915007 | 0.808110 | +0.106897 | 75.39% | 76.56% | +1.17 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.867422 ± 0.069641 | 0.932238 ± 0.062160 | -0.064817 ± 0.097210 | 75.26 ± 0.98 | 76.04 ± 0.90 | +0.78 ± 1.79 |
| delete_0.1 | 0 | 3 | complete | 0.868951 ± 0.062514 | 0.908303 ± 0.095214 | -0.039352 ± 0.119534 | 75.26 ± 1.19 | 75.00 ± 1.95 | -0.26 ± 2.60 |
| delete_0.1 | 1 | 3 | complete | 0.846767 ± 0.035231 | 0.895011 ± 0.060349 | -0.048243 ± 0.092277 | 75.65 ± 1.37 | 76.95 ± 1.70 | +1.30 ± 3.03 |
| delete_0.1 | 2 | 3 | complete | 0.884268 ± 0.037785 | 0.889266 ± 0.070467 | -0.004998 ± 0.102024 | 75.65 ± 0.81 | 74.48 ± 1.85 | -1.17 ± 2.07 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 256 | -0.0570 | - |
| 123 | `fast_slow_contrast` | 256 | +0.1126 | - |
| 123 | `autocorr_tau_ms` | 256 | -0.0355 | - |
| 123 | `active_channel_fraction` | 256 | -0.0583 | - |
| 123 | `duration_ms` | 256 | -0.2054 | - |
| 42 | `mean_rate_hz` | 256 | -0.0440 | - |
| 42 | `fast_slow_contrast` | 256 | +0.0179 | - |
| 42 | `autocorr_tau_ms` | 256 | +0.0710 | - |
| 42 | `active_channel_fraction` | 256 | +0.0243 | - |
| 42 | `duration_ms` | 256 | +0.0637 | - |
| 456 | `mean_rate_hz` | 256 | -0.0444 | - |
| 456 | `fast_slow_contrast` | 256 | +0.1745 | - |
| 456 | `autocorr_tau_ms` | 256 | -0.1827 | - |
| 456 | `active_channel_fraction` | 256 | -0.1665 | - |
| 456 | `duration_ms` | 256 | -0.1409 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors AMT relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
| delete_0.1 | 3 | +0.03395200 +/- 0.00528261 |
