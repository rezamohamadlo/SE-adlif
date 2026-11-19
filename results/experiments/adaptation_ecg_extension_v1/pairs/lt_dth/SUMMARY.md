# ECG extension: LT_DTH_SE_adLIF / DTH_SE_adLIF

All 30 validation rows; seeds 42/123/456; clean, three 10% deletion realizations and 0.9/1.1 time resampling.
Positive benefit means lower variant supervised CE. Repeated realizations are averaged within training seed.
Input, labels and masks share one map; event counts/class proportions can change. No causal interpretation.

| Condition | Accuracy gap mean ± sample SD (pp) | CE benefit mean ± sample SD | Change from clean mean ± sample SD |
|---|---:|---:|---:|
| clean | +0.438125 ± 0.215950 | +0.01344039 ± 0.01580967 | +0.00000000 ± 0.00000000 |
| delete_0.1 | +0.776611 ± 0.546663 | +0.01980180 ± 0.01930035 | +0.00636140 ± 0.01933331 |
| time_scale_0.9 | +0.202106 ± 1.259725 | +0.00978051 ± 0.02819473 | -0.00365988 ± 0.03229816 |
| time_scale_1.1 | +0.613402 ± 0.255790 | +0.01776428 ± 0.00556987 | +0.00432389 ± 0.02107757 |

## Descriptive feature associations

Clean-cohort predictors: fast/slow activity contrast and ACF lag in model steps. Controls: log event rate, duration if nonconstant, and label-class proportions with class 0 omitted.
Status: descriptive_analysis_complete. No patient intervals, p-values or population/causal claims.
Complete-case IDs, explicit exclusions, dropped controls, rank/df, per-seed coefficients and row influence: [ASSOCIATIONS.json](ASSOCIATIONS.json).
The feature windows are 20/200 model steps; they are not established physical milliseconds.

| Clean feature | Coefficient mean ± sample SD across 3 seeds |
|---|---:|
| fast_slow_contrast | -0.03249824 ± 0.04995703 |
| autocorr_tau_steps | -0.00230315 ± 0.01575774 |

Coefficients are CE benefit per population-SD change in that predictor within the fixed cohort.
