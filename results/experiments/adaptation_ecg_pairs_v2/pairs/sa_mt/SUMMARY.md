# ECG SA_MT_SE_adLIF / MT_SE_adLIF

All 30 validation records, seeds 42/123/456. Evaluation only: clean and 10% signed-event deletion.
Positive loss benefit means lower variant CE. Differences in accuracy are percentage points.
Deletion realizations are averaged within each training seed before mean/sample SD over the three seeds.
Original zero-vector argmax-to-class-0 semantics are preserved. No independent-patient or test claim.

| Condition | Seed | Parent validation accuracy (%) | Variant validation accuracy (%) | Gap (pp) | CE benefit | Benefit change |
|---|---:|---:|---:|---:|---:|---:|
| clean | 42 | 86.246477 | 86.031258 | -0.215219 | -0.00172099 | +0.00000000 |
| clean | 123 | 86.307968 | 86.904945 | +0.596977 | +0.01843624 | +0.00000000 |
| clean | 456 | 86.167051 | 86.328465 | +0.161414 | -0.00974566 | +0.00000000 |
| clean | mean ± sample SD (n=3) | 86.240499 ± 0.070649 | 86.421556 ± 0.444220 | +0.181057 ± 0.406454 | +0.00232320 ± 0.01451969 | +0.00000000 ± 0.00000000 |
| delete_0.1 | 42 | 84.504228 | 84.873174 | +0.368947 | +0.00835073 | +0.01007172 |
| delete_0.1 | 123 | 84.710906 | 85.036297 | +0.325391 | +0.01769145 | -0.00074479 |
| delete_0.1 | 456 | 84.298403 | 84.749338 | +0.450935 | +0.00266914 | +0.01241480 |
| delete_0.1 | mean ± sample SD (n=3) | 84.504512 ± 0.206252 | 84.886270 ± 0.143927 | +0.381758 ± 0.063745 | +0.00957044 ± 0.00758506 | +0.00724724 ± 0.00701976 |

Features, label-aligned time scaling and inferential statistics remain pending.
