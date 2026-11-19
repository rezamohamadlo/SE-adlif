# SSC validation mechanism suite

All four evaluation/statistics jobs finished and passed their recorded integrity checks. These are exploratory validation results conditional on three trained seeds, not independent test confirmation. Shared MT predictions are the same observations across comparisons.

| Pair | Parent clean accuracy (%) | Variant clean accuracy (%) | Variant - parent (pp) | Statistics |
|---|---:|---:|---:|---|
| AMT_SE_adLIF / MT_SE_adLIF | 82.1528 +/- 0.2719 | 82.0459 +/- 0.1735 | -0.1069 +/- 0.3889 | [effects and limitations](pairs/amt_mt/statistics_v1/SUMMARY.md) |
| LT_DTH_SE_adLIF / DTH_SE_adLIF | 81.5783 +/- 0.3465 | 81.5917 +/- 0.0504 | 0.0134 +/- 0.3357 | [effects and limitations](pairs/lt_dth/statistics_v1/SUMMARY.md) |
| SA_MT_SE_adLIF / MT_SE_adLIF | 82.1528 +/- 0.2719 | 81.8822 +/- 0.0652 | -0.2705 +/- 0.2768 | [effects and limitations](pairs/sa_mt/statistics_v1/SUMMARY.md) |
| MT_TC_SE_adLIF / MT_SE_adLIF | 82.1528 +/- 0.2719 | 82.1127 +/- 0.1796 | -0.0401 +/- 0.4513 | [effects and limitations](pairs/mt_tc/statistics_v1/SUMMARY.md) |

Values above are copied from each saved clean evaluation summary: mean and sample SD across three matching training seeds. Accuracy SD is in percentage points. No numeric paper baseline is compared here.

Statistical jobs retain exclusions, speaker mappings, shared resampling weights and per-seed effects. Approximate intervals account for eight feature/pair and twelve intervention/pair effects in separate families. Read each direction, seed consistency and limitations; do not convert conditional intervals to causal or independent-confirmation claims.

The portable package includes the exact AMT/MT source snapshot and its PATH_INDEX.json. Historical manifests are not rewritten. Transfer the entire suite directory under the same repository-relative layout.

ECG, stage 7 gradients, direct SSC output-window stability and final scientific synthesis are not completed by this queue.
