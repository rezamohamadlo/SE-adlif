# Four-pair ECG validation evaluation queue

All four pair evaluations finished: 18 clean anchors, 72 unique prediction chunks, 1440 paired rows.
All 30 validation records; seeds 42/123/456; clean plus 10% signed-event deletion (three realizations).
Training precision high, supervised CE, no training or official-test access. Parent predictions are shared.

| Pair | Evaluation | Paired rows | Evidence |
|---|---|---:|---|
| AMT_SE_adLIF / MT_SE_adLIF | complete | 360 | [summary](pairs/amt_mt/SUMMARY.md) |
| LT_DTH_SE_adLIF / DTH_SE_adLIF | complete | 360 | [summary](pairs/lt_dth/SUMMARY.md) |
| SA_MT_SE_adLIF / MT_SE_adLIF | complete | 360 | [summary](pairs/sa_mt/SUMMARY.md) |
| MT_TC_SE_adLIF / MT_SE_adLIF | complete | 360 | [summary](pairs/mt_tc/SUMMARY.md) |

These are descriptive validation evaluations. Full ECG mechanism studies remain incomplete.
Pending: ECG input features, label-aligned time scaling, statistical inference with defensible dependence units, direct SSC window diagnostics, stage 7 gradients.
