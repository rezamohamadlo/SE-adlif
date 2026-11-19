# C005 primary0/3 dropout-consistency four-arm30screen

Fresh training seed42; frozen8/2speaker split; ordinary clean validation only.30/300epochs.

| Arm | Online train in selected epoch | Clean validation | Ordinary validation CE | Selected completed epoch | Speaker0 | Speaker3 |
|---|---:|---:|---:|---:|---:|---:|
| SE_adLIF | 99.64% | 82.14% | 0.9464 | 30 | 82.91% | 81.16% |
| RDROP_SE_adLIF | 98.27% | 83.85% | 0.9443 | 23 | 89.70% | 76.45% |
| RDROP_LS_SE_adLIF | 98.58% | 86.30% | 0.5838 | 22 | 91.13% | 80.19% |
| LS_SE_adLIF | 99.33% | 88.44% | 0.5771 | 28 | 92.88% | 82.83% |

- A_minus_Baseline_pp: +1.712537pp
- B_minus_Baseline_pp: +6.299692pp
- AB_minus_A_pp: +2.446485pp
- AB_minus_B_pp: -2.140671pp
- interaction_pp: -3.853208pp

Tuned development pair,one training seed,30/300epochs; recordings share two speakers. No population/significance/synergy or official-test improvement claim. Fresh dropout-consistency factorial; a positive KL-specific interpretation requires registered alpha0 two-pass controls. Two training graphs add compute; ordinary inference remains one pass. Initialhashmatches do not prove identical later numerical trajectories or significance.

Logged online selected-epoch aggregate with dropout and changing weights; first stochastic view only for RDROP arms, not clean checkpoint training reevaluation. RDROP train loss averages two supervised losses plus alpha times symmetric KL; train_hard_ce is ordinary CE averaged across both views.

Practical threshold:+0.5pp; assess clean accuracy plus CE/learning curves. Smaller gaps alone insufficient. No automatic advancement. No comparable paper validation metric, so no paper delta.
