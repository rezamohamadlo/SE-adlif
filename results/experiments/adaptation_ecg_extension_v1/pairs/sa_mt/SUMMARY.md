# ECG extension: SA_MT_SE_adLIF / MT_SE_adLIF

All 30 validation rows; seeds 42/123/456; clean, three 10% deletion realizations and 0.9/1.1 time resampling.
Positive benefit means lower variant supervised CE. Repeated realizations are averaged within training seed.
Input, labels and masks share one map; event counts/class proportions can change. No causal interpretation.

| Condition | Accuracy gap mean ± sample SD (pp) | CE benefit mean ± sample SD | Change from clean mean ± sample SD |
|---|---:|---:|---:|
| clean | +0.181057 ± 0.406454 | +0.00232320 ± 0.01451969 | +0.00000000 ± 0.00000000 |
| delete_0.1 | +0.381758 ± 0.063745 | +0.00957044 ± 0.00758506 | +0.00724724 ± 0.00701976 |
| time_scale_0.9 | +0.321662 ± 0.693495 | +0.00068818 ± 0.01745271 | -0.00163502 ± 0.02842400 |
| time_scale_1.1 | -0.041929 ± 0.741477 | -0.00409251 ± 0.01187280 | -0.00641571 ± 0.01625420 |

## Descriptive feature associations

Clean-cohort predictors: fast/slow activity contrast and ACF lag in model steps. Controls: log event rate, duration if nonconstant, and label-class proportions with class 0 omitted.
Status: descriptive_analysis_complete. No patient intervals, p-values or population/causal claims.
Complete-case IDs, explicit exclusions, dropped controls, rank/df, per-seed coefficients and row influence: [ASSOCIATIONS.json](ASSOCIATIONS.json).
The feature windows are 20/200 model steps; they are not established physical milliseconds.

| Clean feature | Coefficient mean ± sample SD across 3 seeds |
|---|---:|
| fast_slow_contrast | -0.02343313 ± 0.03765594 |
| autocorr_tau_steps | +0.01204595 ± 0.01942847 |

Coefficients are CE benefit per population-SD change in that predictor within the fixed cohort.
