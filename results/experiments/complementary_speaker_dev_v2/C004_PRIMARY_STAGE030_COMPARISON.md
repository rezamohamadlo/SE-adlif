# C004 primary0/3 current-normalization four-arm30screen

Fresh training seed42; frozen8/2speaker split; ordinary clean validation only.30/300epochs.

| Arm | Online train in selected epoch | Clean validation | Ordinary validation CE | Selected completed epoch | Speaker0 | Speaker3 |
|---|---:|---:|---:|---:|---:|---:|
| SE_adLIF | 98.24% | 83.12% | 0.7628 | 22 | 84.78% | 81.02% |
| RCN_SE_adLIF | 96.31% | 83.00% | 0.7202 | 16 | 86.42% | 78.67% |
| RCN_LS_SE_adLIF | 96.83% | 86.91% | 0.5440 | 16 | 90.69% | 82.13% |
| LS_SE_adLIF | 99.74% | 88.01% | 0.5310 | 30 | 90.58% | 84.76% |

- A_minus_Baseline_pp: -0.122327pp
- B_minus_Baseline_pp: +4.892963pp
- AB_minus_A_pp: +3.914374pp
- AB_minus_B_pp: -1.100916pp
- interaction_pp: -0.978589pp

Tuned development pair,one training seed,30/300epochs; recordings share two speakers. No population/significance/synergy or official-test improvement claim. Fresh current-normalization factorial; normalization-specific attribution requires registered gain controls and learned-scale diagnostics. Gate values alone cannot establish such attribution. Initialhashmatches do not prove identical later numerical trajectories or significance.

Logged online selected-epoch aggregate with dropout and changing weights; not clean checkpoint training reevaluation.

Practical threshold:+0.5pp; assess clean accuracy plus CE/learning curves. Smaller gaps alone insufficient. No automatic advancement. No comparable paper validation metric, so no paper delta.
