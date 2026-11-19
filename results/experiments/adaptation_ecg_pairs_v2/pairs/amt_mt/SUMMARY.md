# ECG AMT_SE_adLIF / MT_SE_adLIF

All 30 validation records, seeds 42/123/456. Evaluation only: clean and 10% signed-event deletion.
Positive loss benefit means lower variant CE. Differences in accuracy are percentage points.
Deletion realizations are averaged within each training seed before mean/sample SD over the three seeds.
Original zero-vector argmax-to-class-0 semantics are preserved. No independent-patient or test claim.

| Condition | Seed | Parent validation accuracy (%) | Variant validation accuracy (%) | Gap (pp) | CE benefit | Benefit change |
|---|---:|---:|---:|---:|---:|---:|
| clean | 42 | 86.246477 | 86.377146 | +0.130669 | -0.00157769 | +0.00000000 |
| clean | 123 | 86.307968 | 86.597489 | +0.289521 | +0.00468688 | +0.00000000 |
| clean | 456 | 86.167051 | 86.723034 | +0.555983 | +0.01555728 | +0.00000000 |
| clean | mean ± sample SD (n=3) | 86.240499 ± 0.070649 | 86.565889 ± 0.175096 | +0.325391 ± 0.214914 | +0.00622216 ± 0.00867004 | +0.00000000 ± 0.00000000 |
| delete_0.1 | 42 | 84.504228 | 84.096849 | -0.407379 | -0.01339519 | -0.01181751 |
| delete_0.1 | 123 | 84.710906 | 83.942267 | -0.768640 | -0.01109402 | -0.01578090 |
| delete_0.1 | 456 | 84.298403 | 85.083269 | +0.784866 | +0.02637424 | +0.01081697 |
| delete_0.1 | mean ± sample SD (n=3) | 84.504512 ± 0.206252 | 84.374128 ± 0.618979 | -0.130384 ± 0.812951 | +0.00062835 ± 0.02232627 | -0.00559381 ± 0.01434965 |

Features, label-aligned time scaling and inferential statistics remain pending.
