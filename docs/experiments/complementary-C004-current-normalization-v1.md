# C004: residual per-frame current normalization

This distinct hypothesis follows the rejected static-threshold branch within
the [active plan](complementary-mechanisms-study-v2.md). C003static-only failed
baseline; static+LS failed LS. Preserve that failure and C002's earlier positive
control rather than selecting whichever cohort looks better.

## Mechanism and prior art

For each sample/frame, compute feed-forward current `c=linear(x,W,b)` and
bias-free component `d=c-b`. Use
`c_new=c+tanh(g)*LayerNorm(d)`, with no learned normalization affine parameters,
epsilon1e-5, and one zero-initialized scalar gate per hidden layer. Statistics
use the last feature axis only, never other samples, past/future frames or
running population statistics. Keep the SE membrane/adaptation/recurrence/reset
and three-state representation unchanged. Raw current remains in the residual;
invariance of the whole model to input amplitude is not claimed.

Zero input gives `d=0`, so the branch is zero even with learned bias. This
preserves the reference feed-forward current on quiet/padded frames without
inferring physical duration from the last event. No sequence warping, truncation,
new mask or changed label/preprocessing is introduced. A zero gate initially
reproduces the reference, while allowing a gate gradient. Normalization may
amplify noise or suppress useful level information; clean accuracy must decide.
Speaker-dependent current variability is a hypothesis, not an established cause.

Normalization is established in [Ba et al., 2016](https://arxiv.org/abs/1607.06450),
zero-gated residuals in [Bachlechner et al., 2020/2021](https://arxiv.org/abs/2003.04887),
and SNN normalization in [Zheng et al., 2020/2021](https://arxiv.org/abs/2011.05280).
No novelty or dynamical-isometry guarantee is claimed. This study applies a
non-affine, bounded residual to feed-forward current with sample/frame-local
statistics. Project SA/VS/AMT and LG-DTH alter adaptation or activity traces;
the targeted source audit found no equivalent registered current-normalization
cell. Prior input gain was static, foldable and not parameter-matched here.

## Matched screen and interpretation

Start fresh Baseline / RCN / RCN+LS / LS, training seed42, primary0/3 frozen
training-only manifest, batch1024, LR0.01, dropout0.15, float32/high precision,
original300epoch schedule and initial30epoch ceiling.75 is conditional on an
explicit matched decision. All inherit the dataset paper baseline. Initial
tensor/RNG hashes remain enabled; no precision/determinism setting changes.
The intended mechanisms alone vary. Training-only smoothing is epsilon0.1;
all validation uses ordinary hard-label CE and clean inputs.

Register gain-only controls before observing results: use identical scalar
parameters and initialization but `c_new=c+tanh(g)*(c-b)`, without normalization.
This per-layer feed-forward gain can be absorbed into weights. The study config
declares plain and smoothed gain controls upfront; run the matched relevant
control if RCN advances, before attributing a benefit to sample-dependent
normalization. These conditional controls are not part of the initial queue.
The gain-only coefficient lies between0and2, whereas the normalized branch's
effective scale also depends on current dispersion. Thus this control alone
cannot prove a normalization-specific cause. Inspect learned effective scales;
if they exceed the control's range, a separately frozen scale-matched follow-up
is needed before that interpretation. This qualification changes no training
formula or selection threshold.

Use clean held-out-speaker accuracy, supported by ordinary validation CE and
curves, with the existing+0.5pp practical gates for A-baseline, AB-A and AB-LS.
Report all contrasts and interaction; no gap-only success or forced synergy.
If unpromising, retain the failure rather than extending flat curves. Only a
frozen primary shortlist can enter6/10 sensitivity or SSC/ECG validation, with
fresh weights and appropriate calibration. Do not tune on the secondary pair.
Official tests stay outside discovery. Canonical SHD* requires later fresh
own-split training, full budgets, matching seeds and a frozen evaluation plan.

Preregistration: `C004_CURRENT_NORMALIZATION_PREREGISTRATION.json` beside the
study results. New outputs use `c004_current_normalization_v1/primary_0_3/`.
Freeze source after non-training checks and discarded-state real-data calibration.
Use the existing runner only:

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF RCN_SE_adLIF RCN_LS_SE_adLIF LS_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_C004_current_norm_v1.yaml --stage-ceiling 30 --dry-run
```

Local Python is `.ven`; use the actual environment name elsewhere. Project
paths are repository-relative; ignored data/checkpoints transfer separately.
