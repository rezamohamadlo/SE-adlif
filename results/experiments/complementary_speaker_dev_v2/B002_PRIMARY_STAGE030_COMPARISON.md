# B002 primary speaker-development screen:30epochs

Ordinary clean validation only; speakers0/3, seed42. Not canonical SHD* or official-test accuracy.

| Arm | Online train accuracy in selected epoch | Best held-out accuracy | Ordinary validation CE | Selected completed epoch | Last10epoch mean accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 98.75% | 81.83% | 0.9221 | 21 | 80.60% |
| A: input gain | 97.25% | 82.32% | 0.8150 | 19 | 79.57% |
| A+B: gain + LS0.1 | 99.41% | 86.73% | 0.6037 | 22 | 85.46% |

A minus Baseline: +0.489295pp. A+B minus A: +4.403669pp.

A+B beats A on clean accuracy and ordinary CE, supported by late curves. A alone is only+0.489pp versus fresh baseline, below+0.5threshold; assess B-only rather than assume complementarity. One historically tuned development pair and one training seed; no final generalization or synergy claim. Epoch-tail SD describes curve fluctuation, not independent-seed uncertainty.

Decision: PROMISING_B002_RUN_B_ONLY_MECHANISM_CONTROL_BEFORE_TRANSFER. Fresh LS_SE_adLIF primary0/3seed42matched30epoch B-only control under unchangedsource, before sensitivity/transfer or expansion.

B training loss is the smoothed objective; hard training CE is separately recorded. Validation CE is ordinary hard-label CE for all arms. No paper delta is applicable to this validation endpoint.

Training accuracy is the logged online epoch aggregate with dropout active and weights changing across batches. It is not a clean full-training-set reevaluation of the fixed selected checkpoint. Validation is a clean fixed-checkpoint pass; the gap alone is not a success criterion.
