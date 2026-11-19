# B002 four-arm primary development screen

Speakers0/3 held out; training seed42;30/300epochs. Clean ordinary validation, not official test or canonical SHD*.

| Arm | Online train accuracy in selected epoch | Best held-out accuracy | Ordinary validation CE | Selected completed epoch | Last10epoch mean accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline | 98.75% | 81.83% | 0.9221 | 21 | 80.60% |
| A: gain | 97.25% | 82.32% | 0.8150 | 19 | 79.57% |
| B: LS0.1 | 99.67% | 86.54% | 0.5944 | 28 | 85.24% |
| A+B: gain+LS0.1 | 99.41% | 86.73% | 0.6037 | 22 | 85.46% |

A_minus_Baseline_pp: +0.489295pp.
B_minus_Baseline_pp: +4.709476pp.
AB_minus_A_pp: +4.403669pp.
AB_minus_B_pp: +0.183487pp.
interaction_pp: -0.305808pp.

Smoothing-only accounts for most gain. AB adds only0.183487pp over B, below+0.5threshold; A alone is0.489295pp over baseline and has worse late mean. Interaction is negative on these validation-selected scores. No complementarity or synergy established. B-only improves held-out accuracy and ordinary CE over baseline; advance only LS-only to fixed-pair diagnostic. Do not extend gain to seek a favorable gap. Single historical development pair and seed, no final generalization claim. Tail-epoch SD is not seed uncertainty.

Next: Fresh Baseline/LS-only fixed6/10pair30epochs; no weights crosspairs, no pair/epsilon tuning. If supportive, SSC/ECG matched validation screens; two-idea goal remains incomplete.

The four-arm development screen is not the final canonical ablation. All original identities/records remain unchanged. B is still B002 in provenance; advancing that component does not retroactively turn it into A002. New complementary ideas require separate preregistration.

Raw metrics, selected checkpoint metadata/hashes and all shared configs verified. Smoothed training objective and ordinary hard-label CE are separately recorded. No comparable published speaker-validation baseline; no paper delta.

Training accuracy is the logged online epoch aggregate with dropout active and weights changing across batches. It is not a clean full-training-set reevaluation of the fixed selected checkpoint. Validation is a clean fixed-checkpoint pass; the gap alone is not a success criterion.
