# B002 smoothing-only SSC40validation comparison

Official validation, seed42, matched batch1024. Both own full-state resume10->40, same original40schedule. No official test results.

| Arm | Online train accuracy in selected epoch | Best validation accuracy | Ordinary validation CE | Selected completed epoch | Last10epoch mean validation |
|---|---:|---:|---:|---:|---:|
| SE_adLIF | 87.46% | 81.52% | 0.6257 | 33 | 80.31% |
| LS_SE_adLIF | 84.88% | 81.96% | 0.6814 | 36 | 80.82% |

LS minus Baseline: +0.430816pp. Practical threshold:+0.5pp.

Nominal validation gain is below preregistered+0.5pp threshold and ordinary validation CE is worse. Preserve single-seed outcome; no success, significance, official-test or uniform cross-dataset improvement claim. Tail-epoch variability is curve fluctuation, not independent-seed uncertainty.

Online epoch aggregate with active dropout and changing weights, not clean fixed-checkpoint training accuracy.

No SSC extension. Verify deferred primary GPU backend replay, then bounded current-objective ECG calibration and fresh matched50epoch validation screen. No heavy new combination or final replication before transfer review.

Raw40unique validation epochs, checkpoint hashes/configs/optimizer/globalstep2920, retained10epoch snapshots and all shared settings verified. No paper delta applies to this validation endpoint.
