# Stage-7 SHD* input-gradient tooling pilot

MT/AMT, seed 42, 40 validation recordings (two per class). No weight training or official test.
Two original-batch full-validation anchors passed. Input gradients and controlled deletions use declared single-record batching.
These are SLAYER surrogate CE sensitivities, not true hard-spike finite differences or causal attention.
Synthetic-padding derivatives and burn-in-input derivatives are recorded separately; padding is not perturbed.

| Model | Nonzero physical gradients | Source-batch prediction differences | Max source-batch CE drift | Max autograd forward CE drift |
|---|---:|---:|---:|---:|
| MT_SE_adLIF | 40 / 40 | 1 | 0.33869958 | 0 |
| AMT_SE_adLIF | 40 / 40 | 0 | 0.00044492126 | 0 |

Shared 1% exact event-deletion budget; MT guidance and three uniform random realizations. Guided/random contrasts use the same eligible records. Predicted linear and actual finite CE changes may disagree; agreement is not a completion gate.

Gradients: 80; evaluation records: 400; paired conditions: 200.
Raw signed arrays: GRADIENT_INDEX.json. Scalar traces and all limits: SUMMARY.json and PAIRED_RECORDS.csv.
Figures: relative normalized physical-input gradient mass, diagnostic/source batch drift, and matched guided/random effects.
This pilot does not complete stage 7. Review numerical behavior and gradient quality before expanding pairs/seeds/data or optimizing inputs.
