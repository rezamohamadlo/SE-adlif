# B002 ECG100: no transfer gain under the declared screen

Seed42; fixed588/30training/validation rows; full-state resume at50 with unchanged400schedule. Validation only, ordinary inference; no official tests.

| Arm | Online train in selected epoch | Validation microaccuracy | Macroclass recall | Ordinary validation CE | Selected completed epoch |
|---|---:|---:|---:|---:|---:|
| SE_adLIF | 79.82% | 77.48% | 73.58% | 0.6399 | 99 |
| LS_SE_adLIF | 79.38% | 76.94% | 72.60% | 0.6827 | 100 |

LS minus baseline: **-0.538045pp**; unchanged practical threshold **+1.0pp**.

LS has lower selected microaccuracy and macrorecall, higher ordinary CE, and no declared practical gain. Preserve promising SHD-only evidence and negative/below-threshold SSC/ECG transfer; do not assume two-dataset benefit or complementarity. Stop this transfer branch at100; no400training or seed expansion on this evidence. This is a staged rejection, not proof no longer-budget benefit is possible.

Online train values are logged epoch aggregates, not clean selected-checkpoint reevaluations. ECG timesteps are correlated within records; patient IDs are unavailable. A single seed and100/400epochs do not establish full-budget or test performance. No compatible paper validation metric is available, so no paper delta.

Both100epoch histories, own best/last hashes/configs/globalstep200, retained optimizer/scheduler, model-source identities and immutable50snapshots verified. Selected-checkpoint inference reproduced saved validation accuracy/CE. See per-seed diagnostics and POST_STAGE_AUDIT.json.
