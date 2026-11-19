# ECG LT_DTH_SE_adLIF / DTH_SE_adLIF

All 30 validation records, seeds 42/123/456. Evaluation only: clean and 10% signed-event deletion.
Positive loss benefit means lower variant CE. Differences in accuracy are percentage points.
Deletion realizations are averaged within each training seed before mean/sample SD over the three seeds.
Original zero-vector argmax-to-class-0 semantics are preserved. No independent-patient or test claim.

| Condition | Seed | Parent validation accuracy (%) | Variant validation accuracy (%) | Gap (pp) | CE benefit | Benefit change |
|---|---:|---:|---:|---:|---:|---:|
| clean | 42 | 85.869844 | 86.512939 | +0.643095 | +0.02483868 | +0.00000000 |
| clean | 123 | 86.290033 | 86.748655 | +0.458622 | +0.02009054 | +0.00000000 |
| clean | 456 | 85.923648 | 86.136305 | +0.212657 | -0.00460804 | +0.00000000 |
| clean | mean ± sample SD (n=3) | 86.027842 ± 0.228653 | 86.465966 ± 0.308865 | +0.438125 ± 0.215950 | +0.01344039 ± 0.01580967 | +0.00000000 ± 0.00000000 |
| delete_0.1 | 42 | 83.561363 | 84.749338 | +1.187975 | +0.04136930 | +0.01653061 |
| delete_0.1 | 123 | 84.210436 | 84.366726 | +0.156290 | +0.00415637 | -0.01593417 |
| delete_0.1 | 456 | 83.877359 | 84.862926 | +0.985567 | +0.01387972 | +0.01848776 |
| delete_0.1 | mean ± sample SD (n=3) | 83.883053 ± 0.324574 | 84.659664 ± 0.259970 | +0.776611 ± 0.546663 | +0.01980180 ± 0.01930035 | +0.00636140 ± 0.01933331 |

Features, label-aligned time scaling and inferential statistics remain pending.
