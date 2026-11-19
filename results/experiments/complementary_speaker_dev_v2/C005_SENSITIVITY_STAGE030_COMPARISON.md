# C005 predefined6/10 sensitivity: clean speaker validation only

One training seed,30/300epochs,predefined6/10 pair with accumulated sensitivity exposure. No pair search,secondary tuning,primary weights reuse,independent population/significance or test improvement claim.

| Arm | Online train at selected epoch | Clean validation | Ordinary CE | Selected completed epoch | Speaker6 | Speaker10 |
|---|---:|---:|---:|---:|---:|---:|
| SE_adLIF | 98.99% | 83.82% | 0.6547 | 27 | 87.70% | 81.62% |
| RDROP_SE_adLIF | 98.29% | 85.52% | 0.6464 | 21 | 86.35% | 85.04% |

A-baseline: +1.699030pp.
Ordinary CE difference: -0.008327.
Last10mean difference: +1.116505pp.

Online changing-weight/dropout aggregate; first stochastic view for RDrop. Training loss averages supervised views plus fixedalpha1 symmetricKL; validation uses one ordinary forward and hard-label CE.

No comparable paper validation metric; no numeric paper delta. No automatic advancement or extension.
