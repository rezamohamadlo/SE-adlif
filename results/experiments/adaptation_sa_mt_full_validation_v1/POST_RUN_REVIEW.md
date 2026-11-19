# SA-MT/MT full-validation output review

Reviewed 12 November 2026 after user execution. Read-only checks confirmed the
completion identity and all 58 recorded output hashes. All 24 imported cache
value hashes (18 MT predictions and six input-feature caches) remain unchanged
from their recorded reuse provenance. The 36 prediction chunks cover the same
1,631 samples in each condition/replicate, with matching checkpoint/seed/cache
identities. All six saved clean anchors are within the specified tolerances.
The agent did not rerun inference, statistical analysis, bootstrap or tests.

The user's completion record reports 156.906 seconds, approximately 2 minutes
37 seconds. This includes reuse of MT predictions and is not an independent
model-latency comparison against previous evaluations.

Saved clean-validation results, mean +/- sample SD over training seeds
42, 123 and 456:

| Metric | MT | SA-MT |
|---|---:|---:|
| Accuracy (%) | 99.2643 +/- 0.0613 | 99.1416 +/- 0.2452 |
| Supervised loss | 0.06651386 +/- 0.02405521 | 0.07081374 +/- 0.02462526 |

Accuracy SD uses percentage points. The paired accuracy gap (SA-MT minus MT)
is -0.1226 +/- 0.2211 pp. SA-MT has lower clean validation accuracy in seeds
42 and 123 and higher accuracy in seed 456. Mean loss benefit (MT minus SA-MT)
is negative. Neither metric establishes a consistent advantage across seeds.
These are validation-development results, not final-test accuracy.

The descriptive intervention tables retain each seed and perturbation
realization. Changes in relative loss benefit are negative on average for
time compression .9, slightly positive for time scaling 1.1 and positive for
10% deletion. Seed variability and the absence of adjusted uncertainty analysis
prevent a significance or causal conclusion from these descriptive values.
Repeated deletion realizations and reused parent predictions are not extra
training runs.

Next: prepare SA-MT-specific adjusted feature-benefit analysis with its own
source identities and transient/sustained-activity hypothesis. Reuse numerical
statistical methods only with explicit assumptions; do not relabel previous
AMT or LT-DTH results. All further experiment execution remains user-launched.
