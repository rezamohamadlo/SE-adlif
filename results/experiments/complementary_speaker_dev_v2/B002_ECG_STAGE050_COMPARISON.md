# B002 ECG50screen: underlearned matched controls

Fresh seed42; fixed588/30row split;50/400epochs; no official tests. Valid timesteps are correlated within recordings and patient IDs are unavailable.

| Arm | Online train in selected epoch | Validation microaccuracy | Macroclass recall | Ordinary validation CE | Selected completed epoch |
|---|---:|---:|---:|---:|---:|
| SE_adLIF | 36.20% | 35.56% | 25.55% | 1.4227 | 50 |
| LS_SE_adLIF | 36.14% | 35.38% | 24.88% | 1.4350 | 50 |

LS minus Baseline: -0.189596pp; preregistered practical threshold:+1.0pp.

LS is nominally worse at50, not a gain. Both best at final50, online train~36% and macroclass recall~25% indicate poorly learned controls, while late accuracy and ordinary CE improve. Extend within predeclared100stage to assess learning, not to claim early generalization. No new B/400run or final replication.

Logged online epoch accuracy with dropout active and changing weights, not clean fixed-checkpoint reevaluation.

Threshold provenance correction: historical A002 used0.1pp; this B002study was preregistered at1.0pp and that numeric rule has not changed.

Exact50epoch histories, checkpoint/config hashes and globalstep100 verified. Selected GPU inference reproduced accuracy/CE; per-class/per-record evidence retained. No comparable paper validation baseline, so no paper delta.
