# ECG stage-6 extension

Executed: 180 input-feature records; 72 preserved and 36 new prediction chunks; 2160 paired records.
No training, official-test access or population inference. All-zero labels retain the original class-0 argmax.
Feature time units are model steps. Time resampling changes input events and label proportions jointly.

| Pair | Paired rows | Descriptive association status | Evidence |
|---|---:|---|---|
| AMT_SE_adLIF / MT_SE_adLIF | 540 | descriptive_analysis_complete | [report](pairs/amt_mt/SUMMARY.md) |
| LT_DTH_SE_adLIF / DTH_SE_adLIF | 540 | descriptive_analysis_complete | [report](pairs/lt_dth/SUMMARY.md) |
| SA_MT_SE_adLIF / MT_SE_adLIF | 540 | descriptive_analysis_complete | [report](pairs/sa_mt/SUMMARY.md) |
| MT_TC_SE_adLIF / MT_SE_adLIF | 540 | descriptive_analysis_complete | [report](pairs/mt_tc/SUMMARY.md) |

Stage 6 remains open for evidence/scope review and the other listed diagnostics.
No patient identities were inferred. Non-estimable designs are reported without replacement models.
