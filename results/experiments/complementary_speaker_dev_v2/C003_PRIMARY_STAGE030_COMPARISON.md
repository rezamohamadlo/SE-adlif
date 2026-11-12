# C003 primary0/3 static four-arm30screen

Fresh training seed42; frozen8/2speaker split; ordinary clean validation only.30/300epochs.

| Arm | Online train in selected epoch | Clean validation | Ordinary validation CE | Selected completed epoch | Speaker0 | Speaker3 |
|---|---:|---:|---:|---:|---:|---:|
| SE_adLIF | 97.92% | 81.65% | 0.9093 | 27 | 86.31% | 75.76% |
| STATIC_SE_adLIF | 98.62% | 78.90% | 0.9471 | 22 | 82.37% | 74.52% |
| STATIC_LS_SE_adLIF | 98.62% | 86.61% | 0.5814 | 22 | 90.25% | 81.99% |
| LS_SE_adLIF | 98.84% | 87.40% | 0.5600 | 27 | 91.35% | 82.41% |

- A_minus_Baseline_pp: -2.752292pp
- B_minus_Baseline_pp: +5.749232pp
- AB_minus_A_pp: +7.706422pp
- AB_minus_B_pp: -0.795102pp
- interaction_pp: +1.957190pp

Tuned development pair,one training seed,30/300epochs; recordings share two speakers. No population/significance/synergy or official-test improvement claim. Fresh staticfactorial; frozenzero activitygains. Initialhashmatches do not prove identical later numerical trajectories or significance.

Logged online selected-epoch aggregate with dropout and changing weights; not clean checkpoint training reevaluation.

Practical threshold:+0.5pp; assess clean accuracy plus CE/learning curves. Smaller gaps alone insufficient. No automatic advancement. No comparable paper validation metric, so no paper delta.
