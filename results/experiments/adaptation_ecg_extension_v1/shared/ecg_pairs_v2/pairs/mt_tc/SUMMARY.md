# ECG MT_TC_SE_adLIF / MT_SE_adLIF

All 30 validation records, seeds 42/123/456. Evaluation only: clean and 10% signed-event deletion.
Positive loss benefit means lower variant CE. Differences in accuracy are percentage points.
Deletion realizations are averaged within each training seed before mean/sample SD over the three seeds.
Original zero-vector argmax-to-class-0 semantics are preserved. No independent-patient or test claim.

| Condition | Seed | Parent validation accuracy (%) | Variant validation accuracy (%) | Gap (pp) | CE benefit | Benefit change |
|---|---:|---:|---:|---:|---:|---:|
| clean | 42 | 86.246477 | 86.077376 | -0.169101 | -0.01847300 | +0.00000000 |
| clean | 123 | 86.307968 | 86.471945 | +0.163976 | +0.00323446 | +0.00000000 |
| clean | 456 | 86.167051 | 85.751986 | -0.415065 | -0.01250901 | +0.00000000 |
| clean | mean ± sample SD (n=3) | 86.240499 ± 0.070649 | 86.100436 ± 0.360533 | -0.140063 ± 0.290611 | -0.00924918 ± 0.01121487 | +0.00000000 ± 0.00000000 |
| delete_0.1 | 42 | 84.504228 | 84.178837 | -0.325391 | -0.01353581 | +0.00493719 |
| delete_0.1 | 123 | 84.710906 | 83.752669 | -0.958237 | -0.02017201 | -0.02340647 |
| delete_0.1 | 456 | 84.298403 | 84.211290 | -0.087112 | +0.00326022 | +0.01576923 |
| delete_0.1 | mean ± sample SD (n=3) | 84.504512 ± 0.206252 | 84.047599 ± 0.255932 | -0.456913 ± 0.450209 | -0.01014920 ± 0.01207763 | -0.00090002 ± 0.02022965 |

Features, label-aligned time scaling and inferential statistics remain pending.
