# B002 candidate note: training-only label smoothing

Status: conditional hypothesis, not implemented, not launched, and not a frozen
experiment registration. A002 SSC transfer and ECG screening remain pending.
Do not change the active matched queue or select this idea using official tests.

## Evidence and mechanism

Primary0/3 development diagnostics show gain-only accuracy improves over its
matched baseline, but selected cross-entropy and confidence on wrong predictions
increase. Brier score improves and ECE evidence is mixed; the result is not
uniformly worse calibration. These observations motivate testing whether a
training objective that discourages extreme class confidence complements input
gain's optimization effect. They do not show that confidence causes its errors,
or that smoothing will correct those errors. A smaller gap or lower loss alone
would not establish success.

Label smoothing is established prior art:
[Szegedy et al., CVPR2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.html)
and [Muller et al., NeurIPS2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html).
The latter reports calibration/generalization benefits in its experiments;
transfer to this SNN and unseen SHD speakers remains an untested hypothesis.
Neither label smoothing nor static input gain is claimed as a new invention.

## Conditional bounded experiment

Advance to an actual registration only after reviewing A002 transfer evidence.
If A002 fails transfer, investigate/revise A before substantial B investment.
An honest negative result remains acceptable; do not force a complementary pair.

Proposed initial coefficient is epsilon0.1, fixed from established practice,
not searched on either speaker pair. Training target distribution would be
`q=(1-epsilon)*one_hot(y)+epsilon/C`. Apply it to the established target-aligned
temporal aggregates, preserving each dataset's aggregation, masks, target dtype,
sequence length and labels. Validation/test retain ordinary hard-target CE and
unchanged inference. Regression must reject this classification-only objective.

Use a separate explicit opt-in objective and fresh versioned output paths via
the existing runner. Because factory dispatch currently rejects FIR/gain plus
objective variants, any implementation must explicitly register and test this
combination rather than silently replacing ordinary model behavior. The
baseline, A and A+B must be freshly matched under that source/config version
unless exact historical compatibility is proven. No exposed speaker weights
may initialize a held-out-speaker screen.

Primary development would use frozen0/3, seed42,30epochs and the same matched
batch/LR/full scheduler as its controls. Primary decision: clean validation
accuracy A+B minus A, proposed practical threshold+0.5pp, supported by ordinary
validation CE and learning curves. No improvement claim from ECE, Brier, gap or
training accuracy alone. Predeclare the extension rule and compute estimate
before launch; only shortlisted methods enter the existing fixed6/10 diagnostic.
Do not tune epsilon on6/10 or search for a better pair. B-only remains mandatory
for the final factorial comparison and may be required earlier to resolve
whether an observed gain is generic regularization rather than complementarity.

## Required verification before implementation/launch

- Confirm alpha0 is exactly ordinary CE, including masking and finite gradients.
- Validate against an independent explicit soft-target CE calculation on CPU
  and CUDA when available, without fitting a model as a unit test.
- Verify train-only application: evaluation loss/predictions remain ordinary,
  with no extra inference parameters, TTA, post-hoc probability fitting or
  inference-time rescaling. Evaluation identity must be recorded in metadata.
- Keep separate objective loss and hard-target training CE if both are logged;
  never compare smoothed training loss numerically with unsmoothed validation CE
  as though they were the same objective.
- Preserve classification-only target conversion, ignored targets, ECG end
  markers and per-timestep semantics. Report class-sensitive ECG diagnostics:
  smoothing may hurt minority/boundary classes despite aggregate accuracy.
- Record objective identity/deviations in resolved configs, stage receipts,
  source/checkpoint guards and final frozen arm mapping. Reject incompatible
  checkpoint reuse and retain all unsuccessful evidence.

No source/config/test file was changed to create this note. It authorizes no
claim of synergy, novelty, two-dataset improvement or official-test benefit.
