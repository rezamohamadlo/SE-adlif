# Fixed-precision SHD* input-gradient repeat pilot

MT/AMT, seed 42, same 40 validation recordings. No training, official test, event deletion or input optimization.
Two full-validation references passed under highest_matmul; these do not replace source_high training-validation anchors.
Saved gradients: 164; repeat/padding checks: 86; cross-precision legacy comparisons: 80.
Declared stability criteria passed: True. Autograd-forward failures: 0; strict reference failures: 0.
Zero physical gradients are explicit; no normalized maps or cosine values are fabricated for them.
All model state hashes unchanged. Criteria are engineering thresholds, not scientific significance or causal attribution.
Completion authenticates negative results too. Review SUMMARY.json, STABILITY_CHECKS.csv and raw GRADIENT_INDEX.json before expansion.
Source-high historical arrays, same-mode reference subset and portable path index are preserved under shared/.
Stage 7 remains incomplete. No event-deletion or optimized-input queue launches automatically.
