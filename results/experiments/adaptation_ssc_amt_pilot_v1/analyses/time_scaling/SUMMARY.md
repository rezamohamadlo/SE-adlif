# SSC paired validation pilot: time_scaling

**Pilot subset: 256 fixed validation records.** Full clean anchors use 9981 records per checkpoint and are stored separately in `../../VALIDATION_ANCHORS.json`. These tables and feature associations describe the subset only.

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 256; paired records across all groups: 2304.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 256 | 0.888001 | 0.999812 | -0.111811 | 76.17% | 75.00% | -1.17 pp |
| 123 | time_scale_0.9 | 0 | 256 | 1.004292 | 1.009859 | -0.005567 | 74.22% | 71.09% | -3.12 pp |
| 123 | time_scale_1.1 | 0 | 256 | 0.971858 | 1.017446 | -0.045588 | 75.39% | 74.22% | -1.17 pp |
| 42 | clean | 0 | 256 | 0.789810 | 0.919410 | -0.129600 | 75.39% | 76.56% | +1.17 pp |
| 42 | time_scale_0.9 | 0 | 256 | 0.922913 | 0.999205 | -0.076292 | 73.44% | 73.44% | +0.00 pp |
| 42 | time_scale_1.1 | 0 | 256 | 0.863851 | 0.957821 | -0.093970 | 76.95% | 75.78% | -1.17 pp |
| 456 | clean | 0 | 256 | 0.924454 | 0.877493 | +0.046961 | 74.22% | 76.56% | +2.34 pp |
| 456 | time_scale_0.9 | 0 | 256 | 0.987001 | 0.923810 | +0.063191 | 72.27% | 75.00% | +2.73 pp |
| 456 | time_scale_1.1 | 0 | 256 | 0.917072 | 0.899032 | +0.018040 | 75.00% | 74.22% | -0.78 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.867422 ± 0.069641 | 0.932238 ± 0.062160 | -0.064817 ± 0.097210 | 75.26 ± 0.98 | 76.04 ± 0.90 | +0.78 ± 1.79 |
| time_scale_0.9 | 0 | 3 | complete | 0.971402 ± 0.042873 | 0.977625 ± 0.046909 | -0.006223 ± 0.069744 | 73.31 ± 0.98 | 73.18 ± 1.97 | -0.13 ± 2.93 |
| time_scale_1.1 | 0 | 3 | complete | 0.917594 ± 0.054006 | 0.958100 ± 0.059208 | -0.040506 ± 0.056178 | 75.78 ± 1.03 | 74.74 ± 0.90 | -1.04 ± 0.23 |

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
| time_scale_0.9 | 3 | +0.05859402 +/- 0.04523906 |
| time_scale_1.1 | 3 | +0.02431034 +/- 0.04857154 |
