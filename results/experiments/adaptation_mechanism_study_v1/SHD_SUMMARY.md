# SHD* mechanism study: four-pair consolidation

Reviewed 12 November 2026. This document copies saved full-validation aggregates and interval summaries; no new experiment, fit, resampling or tests were run. `SHD_SOURCES.json` records completion identities and verified artifact hashes.

## Cohort and interpretation

All four comparisons use the same 1,631 validation records, split seed 42 and training seeds 42/123/456. Values are validation-development results, not independent final-test scores. Ordinary inference, no TTA. Paper-baseline metadata is paper_classification_v1; MT-TC intentionally changes its training objective. No numeric paper-test baseline is compared to these incompatible validation metrics.

MT is shared by AMT, SA-MT and MT-TC; DTH is LT-DTH's parent. These four comparisons do not have four independent baseline cohorts. There are six model types and eighteen selected model/seed checkpoints. Reused predictions and perturbation realizations do not create independent training runs. No joint cross-model significance test or ranking is claimed.

## Clean validation performance

Accuracy values are percentages; SD and accuracy gaps use percentage points. Each gap is variant minus its own parent, copied from the saved paired aggregate, not a difference of SDs.

| Pair | Parent accuracy (mean +/- sample SD) | Variant accuracy (mean +/- sample SD) | Paired accuracy gap (pp, mean +/- sample SD) | Loss benefit (mean +/- sample SD) |
|---|---:|---:|---:|---:|
| AMT / MT | 99.2643 +/- 0.0613 | 99.0190 +/- 0.1062 | -0.2452 +/- 0.1226 | -0.00734363 +/- 0.03660202 |
| LT-DTH / DTH | 99.0803 +/- 0.0613 | 99.1416 +/- 0.3186 | +0.0613 +/- 0.3729 | +0.00829592 +/- 0.01570407 |
| SA-MT / MT | 99.2643 +/- 0.0613 | 99.1416 +/- 0.2452 | -0.1226 +/- 0.2211 | -0.00429988 +/- 0.00734468 |
| MT-TC / MT | 99.2643 +/- 0.0613 | 99.1825 +/- 0.0937 | -0.0817 +/- 0.0937 | -0.00207534 +/- 0.01294552 |

Positive loss benefit = supervised loss(parent) - supervised loss(variant). MT-TC auxiliary training JS is excluded. Clean accuracy does not establish a consistent improvement across all three seeds for any variant. Accuracy and loss can rank a pair differently.

## Matching training seeds

| Pair | Seed | Parent accuracy (%) | Variant accuracy (%) | Accuracy gap (pp) | Loss benefit |
|---|---:|---:|---:|---:|---:|
| AMT / MT | 42 | 99.2643 | 98.8964 | -0.3679 | +0.02149747 |
| AMT / MT | 123 | 99.2029 | 99.0803 | -0.1226 | +0.00499120 |
| AMT / MT | 456 | 99.3256 | 99.0803 | -0.2452 | -0.04851957 |
| LT-DTH / DTH | 42 | 99.1416 | 98.9577 | -0.1839 | -0.00594974 |
| LT-DTH / DTH | 123 | 99.0803 | 98.9577 | -0.1226 | +0.00570210 |
| LT-DTH / DTH | 456 | 99.0190 | 99.5095 | +0.4905 | +0.02513542 |
| SA-MT / MT | 42 | 99.2643 | 98.8964 | -0.3679 | -0.00221566 |
| SA-MT / MT | 123 | 99.2029 | 99.1416 | -0.0613 | -0.01246143 |
| SA-MT / MT | 456 | 99.3256 | 99.3869 | +0.0613 | +0.00177744 |
| MT-TC / MT | 42 | 99.2643 | 99.0803 | -0.1839 | +0.01275690 |
| MT-TC / MT | 123 | 99.2029 | 99.2029 | +0.0000 | -0.00788211 |
| MT-TC / MT | 456 | 99.3256 | 99.2643 | -0.0613 | -0.01110081 |

## Adjusted input-feature associations

Both features enter jointly, with class/speaker effects and log duration/activity controls. Coefficients use one original-cohort feature SD. Brackets are saved approximate Bonferroni-adjusted 97.5% speaker-bootstrap intervals for a two-feature family within each pair. They are not a simultaneous family across four pairs.

| Pair | Fast/slow coefficient [interval] | Input ACF coefficient [interval] | Interpretation |
|---|---:|---:|---|
| AMT / MT | +0.004764 [-0.026625, +0.054401] | +0.026769 [+0.003042, +0.063770] | Positive ACF within fixed cohort; seed directions differ |
| LT-DTH / DTH | -0.010032 [-0.061780, +0.024167] | -0.014033 [-0.053943, +0.008900] | Both inconclusive |
| SA-MT / MT | -0.019664 [-0.047620, +0.001216] | +0.032337 [+0.005389, +0.078847] | Positive ACF within fixed cohort; seed directions differ |
| MT-TC / MT | +0.012434 [-0.056932, +0.078558] | +0.030419 [-0.008861, +0.111900] | Both inconclusive |

Each analysis has three fixed training seeds and ten speakers, 1,631 complete cases and 2,000 valid draws. Bootstrap coverage may be poor with ten clusters; intervals do not establish robustness over a population of training seeds, causality or independent confirmation. Specifications were written after descriptive validation results were observed. No formal p-values are reported. Positive fixed-cohort ACF intervals for AMT and SA-MT have conflicting seed directions. Negative/mixed findings remain evidence; feature selection was not changed to obtain positive results.

## Controlled perturbations

Entries copy mean change in loss benefit and its approximate corrected interval; positive favors the variant relative to clean input. A separate three-intervention family is used within each pair. Deletion realizations are averaged within sample/seed first.

| Pair | Time scale .9 | Time scale 1.1 | Deletion .1 |
|---|---:|---:|---:|
| AMT / MT | -0.008728 [-0.032091, +0.007219] | +0.002114 [-0.021878, +0.026025] | +0.001476 [-0.006908, +0.011398] |
| LT-DTH / DTH | -0.011365 [-0.037136, +0.015769] | -0.002117 [-0.015867, +0.012959] | -0.005778 [-0.013320, -0.000443] |
| SA-MT / MT | -0.011930 [-0.036516, +0.004683] | +0.000501 [-0.021833, +0.018629] | +0.002337 [-0.009266, +0.011061] |
| MT-TC / MT | -0.023025 [-0.046182, -0.002106] | +0.009062 [-0.007546, +0.028551] | +0.008223 [-0.003511, +0.021013] |

LT-DTH deletion and MT-TC compression .9 have exploratory negative interval directions. Only MT-TC compression has negative estimates in all three observed seeds. These are fixed-cohort results with limited cluster coverage, not universal harms or formal causal confirmation. Warping changes duration/rate/structure together; deletion changes several input statistics together.

## Direct MT-TC output-window evidence

The clean 20-ms nonoverlapping-window diagnostic has its own mask excluding burn-in and padding; ordinary prediction compatibility gates passed. MT-TC has lower adjacent-window JS in seed 42, but higher JS in 123 and 456. Higher mean true-class mass in all seeds did not produce consistently higher window accuracy. Stable wrong/low-confidence evidence is not useful evidence. These descriptive output metrics do not support a general claim that MT-TC yields smoother, better temporal decisions. No confidence-to-benefit regression or mediation claim was fitted.

## Evidence links

- AMT / MT: [evaluation](../adaptation_mechanism_full_validation_v1/analyses/clean/SUMMARY.md), [adjusted statistics](../adaptation_mechanism_full_validation_v1/statistics_v1/SUMMARY.md).
- LT-DTH / DTH: [evaluation](../adaptation_lt_dth_full_validation_v1/analyses/clean/SUMMARY.md), [adjusted statistics](../adaptation_lt_dth_full_validation_v1/statistics_v1/SUMMARY.md).
- SA-MT / MT: [evaluation](../adaptation_sa_mt_full_validation_v1/analyses/clean/SUMMARY.md), [adjusted statistics](../adaptation_sa_mt_full_validation_v1/statistics_v1/SUMMARY.md).
- MT-TC / MT: [evaluation](../adaptation_mt_tc_full_validation_v1/analyses/clean/SUMMARY.md), [adjusted statistics](../adaptation_mt_tc_full_validation_v1/statistics_v1/SUMMARY.md).
- MT-TC: [window evidence](../adaptation_mt_tc_full_validation_v1/window_evidence_v1/SUMMARY.md), [joint interpretation](../adaptation_mt_tc_full_validation_v1/MT_TC_ANALYSIS_REVIEW.md).

## Remaining roadmap work

SHD* is one dataset. This consolidation does not finish SSC/ECG, the full paper-analysis code map, cross-dataset feature distributions, gradient/input-optimization analyses or the final computational-cost/mechanism report. Existing training costs are not new perturbation-specific latency or energy measurements.

Next: inspect SSC prerequisites independently and prepare a bounded validation-only pilot before any full mechanism study. SSC uses official validation, mean-membrane prediction loss and its dataset-specific baseline; SHD split/budget settings must not be copied. All scientific execution and tests are user-launched.
