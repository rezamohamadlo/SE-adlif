# ECG extension: AMT_SE_adLIF / MT_SE_adLIF

All 30 validation rows; seeds 42/123/456; clean, three 10% deletion realizations and 0.9/1.1 time resampling.
Positive benefit means lower variant supervised CE. Repeated realizations are averaged within training seed.
Input, labels and masks share one map; event counts/class proportions can change. No causal interpretation.

| Condition | Accuracy gap mean ± sample SD (pp) | CE benefit mean ± sample SD | Change from clean mean ± sample SD |
|---|---:|---:|---:|
| clean | +0.325391 ± 0.214914 | +0.00622216 ± 0.00867004 | +0.00000000 ± 0.00000000 |
| delete_0.1 | -0.130384 ± 0.812951 | +0.00062835 ± 0.02232627 | -0.00559381 ± 0.01434965 |
| time_scale_0.9 | +0.363412 ± 0.879633 | +0.01283789 ± 0.02946263 | +0.00661573 ± 0.02130957 |
| time_scale_1.1 | +0.201103 ± 0.164206 | +0.00857843 ± 0.00344758 | +0.00235627 ± 0.01205833 |

## Descriptive feature associations

Clean-cohort predictors: fast/slow activity contrast and ACF lag in model steps. Controls: log event rate, duration if nonconstant, and label-class proportions with class 0 omitted.
Status: descriptive_analysis_complete. No patient intervals, p-values or population/causal claims.
Complete-case IDs, explicit exclusions, dropped controls, rank/df, per-seed coefficients and row influence: [ASSOCIATIONS.json](ASSOCIATIONS.json).
The feature windows are 20/200 model steps; they are not established physical milliseconds.

| Clean feature | Coefficient mean ± sample SD across 3 seeds |
|---|---:|
| fast_slow_contrast | +0.01485280 ± 0.03720231 |
| autocorr_tau_steps | +0.00238823 ± 0.01345194 |

Coefficients are CE benefit per population-SD change in that predictor within the fixed cohort.
