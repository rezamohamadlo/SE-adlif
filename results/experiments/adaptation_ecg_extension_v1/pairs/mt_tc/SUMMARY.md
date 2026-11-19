# ECG extension: MT_TC_SE_adLIF / MT_SE_adLIF

All 30 validation rows; seeds 42/123/456; clean, three 10% deletion realizations and 0.9/1.1 time resampling.
Positive benefit means lower variant supervised CE. Repeated realizations are averaged within training seed.
Input, labels and masks share one map; event counts/class proportions can change. No causal interpretation.

| Condition | Accuracy gap mean ± sample SD (pp) | CE benefit mean ± sample SD | Change from clean mean ± sample SD |
|---|---:|---:|---:|
| clean | -0.140063 ± 0.290611 | -0.00924918 ± 0.01121487 | +0.00000000 ± 0.00000000 |
| delete_0.1 | -0.456913 ± 0.450209 | -0.01014920 ± 0.01207763 | -0.00090002 ± 0.02022965 |
| time_scale_0.9 | +0.037005 ± 0.387918 | +0.00018669 ± 0.01986270 | +0.00943587 ± 0.01384603 |
| time_scale_1.1 | -0.354065 ± 0.198161 | -0.01240109 ± 0.00690432 | -0.00315190 ± 0.01678405 |

## Descriptive feature associations

Clean-cohort predictors: fast/slow activity contrast and ACF lag in model steps. Controls: log event rate, duration if nonconstant, and label-class proportions with class 0 omitted.
Status: descriptive_analysis_complete. No patient intervals, p-values or population/causal claims.
Complete-case IDs, explicit exclusions, dropped controls, rank/df, per-seed coefficients and row influence: [ASSOCIATIONS.json](ASSOCIATIONS.json).
The feature windows are 20/200 model steps; they are not established physical milliseconds.

| Clean feature | Coefficient mean ± sample SD across 3 seeds |
|---|---:|
| fast_slow_contrast | -0.03926368 ± 0.04039310 |
| autocorr_tau_steps | +0.01872644 ± 0.01073528 |

Coefficients are CE benefit per population-SD change in that predictor within the fixed cohort.
