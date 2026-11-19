# C001 primary0/3 four-arm30screen

Fresh training seed42; frozen8/2speaker split; ordinary clean validation only.30/300epochs.

| Arm | Online train in selected epoch | Clean validation | Ordinary validation CE | Selected completed epoch | Speaker0 | Speaker3 |
|---|---:|---:|---:|---:|---:|---:|
| SE_adLIF | 99.43% | 83.30% | 0.9106 | 30 | 87.73% | 77.70% |
| DTH_SE_adLIF | 98.96% | 80.98% | 1.0900 | 28 | 85.32% | 75.48% |
| DTH_LS_SE_adLIF | 97.95% | 86.91% | 0.5627 | 22 | 91.02% | 81.72% |
| LS_SE_adLIF | 99.45% | 85.38% | 0.6307 | 27 | 91.79% | 77.29% |

- A_minus_Baseline_pp: -2.324158pp
- B_minus_Baseline_pp: +2.079511pp
- AB_minus_A_pp: +5.932719pp
- AB_minus_B_pp: +1.529050pp
- interaction_pp: +3.853208pp

Tuned development pair,one training seed,30/300epochs; recordings share two speakers. No population/significance/synergy or official-test improvement claim. DTH parameter inspection does not isolate static versus dynamic contribution.

Logged online selected-epoch aggregate with dropout and changing weights; not clean checkpoint training reevaluation.

Practical threshold:+0.5pp; assess clean accuracy plus CE/learning curves. Smaller gaps alone insufficient. No automatic advancement. No comparable paper validation metric, so no paper delta.
