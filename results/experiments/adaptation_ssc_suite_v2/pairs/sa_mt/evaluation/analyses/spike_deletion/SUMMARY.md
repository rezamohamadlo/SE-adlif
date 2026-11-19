# SSC paired full validation: SA_MT_SE_adLIF versus MT_SE_adLIF / spike_deletion

**Full official SSC validation: 9981 records per condition.** Clean results also supply the saved compatibility anchors. These are exploratory validation diagnostics, with the original class frequencies. They are not independent confirmation or final-test results.

Frozen-checkpoint, ordinary-inference validation analysis. These are not final-test results and do not select an architecture. No test-time augmentation is used.

Loss benefit is parent loss minus variant loss; positive values favor the variant. Accuracy differences are variant minus parent in percentage points (pp). Each seed, condition and perturbation replicate is summarized separately; repeated samples and seeds are not pooled as independent observations.

Unique validation samples: 9981; paired records across all groups: 119772.

| Seed | Condition | Replicate | Validation samples | Parent loss | Variant loss | Loss benefit | Parent accuracy | Variant accuracy | Accuracy difference |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 123 | clean | 0 | 9981 | 0.640417 | 0.632473 | +0.007944 | 82.00% | 81.95% | -0.05 pp |
| 123 | delete_0.1 | 0 | 9981 | 0.652439 | 0.644012 | +0.008427 | 81.73% | 81.63% | -0.10 pp |
| 123 | delete_0.1 | 1 | 9981 | 0.647827 | 0.641609 | +0.006217 | 81.52% | 81.30% | -0.22 pp |
| 123 | delete_0.1 | 2 | 9981 | 0.654859 | 0.643476 | +0.011383 | 81.51% | 81.39% | -0.12 pp |
| 42 | clean | 0 | 9981 | 0.612064 | 0.613298 | -0.001234 | 82.00% | 81.82% | -0.18 pp |
| 42 | delete_0.1 | 0 | 9981 | 0.643459 | 0.640099 | +0.003360 | 81.06% | 80.84% | -0.22 pp |
| 42 | delete_0.1 | 1 | 9981 | 0.639903 | 0.637597 | +0.002306 | 81.00% | 81.08% | +0.08 pp |
| 42 | delete_0.1 | 2 | 9981 | 0.645546 | 0.640730 | +0.004816 | 80.72% | 80.84% | +0.12 pp |
| 456 | clean | 0 | 9981 | 0.612423 | 0.636652 | -0.024229 | 82.47% | 81.89% | -0.58 pp |
| 456 | delete_0.1 | 0 | 9981 | 0.631872 | 0.676832 | -0.044960 | 81.34% | 80.18% | -1.16 pp |
| 456 | delete_0.1 | 1 | 9981 | 0.621467 | 0.664308 | -0.042841 | 81.94% | 80.74% | -1.19 pp |
| 456 | delete_0.1 | 2 | 9981 | 0.634879 | 0.667771 | -0.032891 | 81.44% | 80.76% | -0.68 pp |

## Across matching training seeds

Each entry is the mean ± sample SD (ddof=1) of the per-seed values above, with equal weight per training seed. Conditions and perturbation replicates remain separate. SD describes variation across training seeds, not a confidence interval; it is undefined when n=1. Accuracy means are percentages and their SDs are percentage points; accuracy differences and their SDs are also in pp. A partial cohort lists missing seeds and is not a complete-cohort comparison.

| Condition | Replicate | Training seeds (n) | Cohort | Parent loss | Variant loss | Loss benefit | Parent accuracy (%) | Variant accuracy (%) | Accuracy difference (pp) |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| clean | 0 | 3 | complete | 0.621635 ± 0.016267 | 0.627474 ± 0.012454 | -0.005840 ± 0.016573 | 82.15 ± 0.27 | 81.88 ± 0.07 | -0.27 ± 0.28 |
| delete_0.1 | 0 | 3 | complete | 0.642590 ± 0.010311 | 0.653648 ± 0.020173 | -0.011058 ± 0.029469 | 81.38 ± 0.33 | 80.88 ± 0.72 | -0.49 ± 0.58 |
| delete_0.1 | 1 | 3 | complete | 0.636399 ± 0.013525 | 0.647838 ± 0.014404 | -0.011439 ± 0.027265 | 81.49 ± 0.47 | 81.04 ± 0.28 | -0.44 ± 0.67 |
| delete_0.1 | 2 | 3 | complete | 0.645094 ± 0.009997 | 0.650659 ± 0.014883 | -0.005564 ± 0.023893 | 81.23 ± 0.44 | 81.00 ± 0.34 | -0.23 ± 0.41 |

## Descriptive clean-input feature associations

Pearson r is calculated separately for each seed using finite pairs only, with at least three pairs and nonconstant feature and benefit. No p-values or causal claims are made. Undefined values are retained explicitly.

| Seed | Feature | Finite pairs | Pearson r | Undefined reason |
|---|---|---:|---:|---|
| 123 | `mean_rate_hz` | 9981 | -0.0244 | - |
| 123 | `fast_slow_contrast` | 9981 | +0.0187 | - |
| 123 | `autocorr_tau_ms` | 9979 | -0.0162 | - |
| 123 | `active_channel_fraction` | 9981 | -0.0154 | - |
| 123 | `duration_ms` | 9981 | +0.0043 | - |
| 42 | `mean_rate_hz` | 9981 | +0.0215 | - |
| 42 | `fast_slow_contrast` | 9981 | -0.0167 | - |
| 42 | `autocorr_tau_ms` | 9979 | +0.0158 | - |
| 42 | `active_channel_fraction` | 9981 | -0.0135 | - |
| 42 | `duration_ms` | 9981 | -0.0166 | - |
| 456 | `mean_rate_hz` | 9981 | -0.0334 | - |
| 456 | `fast_slow_contrast` | 9981 | +0.0127 | - |
| 456 | `autocorr_tau_ms` | 9979 | +0.0084 | - |
| 456 | `active_channel_fraction` | 9981 | +0.0042 | - |
| 456 | `duration_ms` | 9981 | +0.0007 | - |

Figures are descriptive. Conditions are categorical; connecting lines do not imply a continuous dose response. Checkpoint, split and feature provenance are retained in the accompanying `MANIFEST.json`; group and environment metadata are retained in `SUMMARY.json`.

## Change from clean input

Perturbation realizations are averaged within each training seed first. Across-seed values are mean and sample SD; three seeds are not a confidence interval. Positive loss-benefit change favors SA_MT_SE_adLIF relative to clean input. No causal claim or significance test is made.

| Condition | Training seeds | Benefit change (mean +/- SD) |
|---|---:|---:|
| clean | 3 | +0.00000000 +/- 0.00000000 |
| delete_0.1 | 3 | -0.00351414 +/- 0.01099799 |
