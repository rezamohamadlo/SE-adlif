# C005 alpha0 control: clean speaker validation only

| Arm | Validation accuracy | Ordinary CE | Selected completed epoch |
|---|---:|---:|---:|
| SE_adLIF | 82.14% | 0.9464 | 30 |
| RDROP_SE_adLIF | 83.85% | 0.9443 | 23 |
| TWO_PASS_SE_adLIF | 81.04% | 1.0508 | 29 |

RDrop minus alpha0 control: +2.813456pp.
Alpha0 control minus baseline: -1.100917pp.

One seed,tuned primary0/3,30/300epochs; opposite speaker effects and repeated-seed variability require care. This control isolates KL from two-forward supervision; no automatic advancement, population significance, synergy, test or paper improvement claim.
