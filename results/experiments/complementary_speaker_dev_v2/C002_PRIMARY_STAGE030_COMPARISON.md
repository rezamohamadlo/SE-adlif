# C002 bounded static versus dynamic threshold control

All3fresh with training-only smoothing epsilon0.1; frozen primary0/3; seed42;30/300epochs.

| Arm | Online train in selected epoch | Clean validation | Ordinary validation CE | Selected completed epoch | Speaker0 | Speaker3 |
|---|---:|---:|---:|---:|---:|---:|
| LS_SE_adLIF | 99.50% | 86.73% | 0.5795 | 29 | 90.36% | 82.13% |
| STATIC_LS_SE_adLIF | 99.30% | 87.77% | 0.5672 | 25 | 91.24% | 83.38% |
| DTH_LS_SE_adLIF | 99.28% | 87.03% | 0.6141 | 25 | 90.25% | 82.96% |

- static_minus_LS_pp: +1.039755pp
- dynamic_minus_LS_pp: +0.305814pp
- dynamic_minus_static_pp: -0.733942pp

Primary0/3development has extensive tuning exposure,one training seed,30/300epochs. Static arm retains trace computations; no inference-speed claim. This mechanism control does not replace the final four-arm ablation. No secondary tuning, causality proof, population or official-test claim.

Logged online selected-epoch aggregate with dropout and changing weights; not clean checkpoint training reevaluation.

No comparable paper validation metric, so no paper delta. Review original+0.5pp threshold with ordinary CE/curves. No automatic next job.
