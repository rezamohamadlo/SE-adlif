# B002 smoothing-only: predefined pair sensitivity

Validation speakers6/10; fresh seed42models,30/300epochs. No official tests.

| Arm | Online train accuracy in selected epoch | Held-out accuracy | Ordinary validation CE | Selected completed epoch | Last10epoch mean accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 99.41% | 84.30% | 0.6968 | 30 | 82.73% |
| LS0.1 | 99.45% | 89.48% | 0.5316 | 26 | 87.59% |

LS minus Baseline: +5.177993pp.

LS-only improves clean accuracy and ordinary validation CE on both predefined pairs; shortlist component for cross-dataset checks, no pair/epsilon tuning. This is one training seed, previously observed training-speaker pairs, short30epoch budgets and validation-selected peaks; not independent final-test or population evidence. Gain+LS is not established as complementary. Two-idea goal remains incomplete.

Disposable SSC current-source/LS-objective batch1024 and validation parity confirmation, discard states; then fresh matched Baseline/LS-only SSC10screen with original40schedule.

Shared config, checkpoint metadata/hashes, all30raw epochs and archived source verified. No paper delta applies to speaker-validation scores.

Training accuracy is the logged online epoch aggregate with dropout active and weights changing across batches. It is not a clean full-training-set reevaluation of the fixed selected checkpoint. Validation is a clean fixed-checkpoint pass; the gap alone is not a success criterion.
