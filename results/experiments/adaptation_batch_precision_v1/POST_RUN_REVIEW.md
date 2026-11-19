# SHD* batching/precision diagnostic: saved-output review

Reviewed 16 November 2026. **Diagnostic completed and verified. Stage 7 remains incomplete.** No agent tests, inference, gradients, resampling or fresh experiment execution occurred during this review.

Authenticated **568 registered outputs**, **57 scientific source hashes**, both checkpoint file hashes and the **16 passed user contracts**. The 546 digest-guarded jobs contain 9,786 full-validation score rows, 720 pilot layout rows and 24 focus control rows. All 1,240 saved comparisons agree with arithmetic checks of scalar/logit differences. Original high-precision anchors passed unchanged tolerances; all saved model-state checks are unchanged, with no accumulated parameter gradients. The original pilot remains preserved.

## What the controls show

For MT recording **3896**, `source_high` original batching predicts **1 (correct)** with CE **0.5342804193**; its natural singleton predicts **9 (incorrect)** with CE **0.8729850054**, a difference of **+0.3387045860**. Retaining the original 295-step padding still predicts 9. Repeating the original batch, replacing the 256-record batch with copies of the target, or reversing original batch order still predicts 1. Padding length and those tested companion/order changes therefore do not explain away this focal discrepancy.

With `highest_matmul` (matmul TF32 disabled, cuDNN TF32 retained), all tested focal layouts predict 9, including original batching. Across the 40 pilot recordings, both stricter modes have **zero batch-layout prediction mismatches** for either model. Maximum singleton/original CE drift is **6.6161e-6** for MT in `highest_matmul`, **5.5432e-6** with cuDNN TF32 also disabled, and **zero** for AMT. Disabling cuDNN TF32 adds no prediction change in this matrix.

These controlled observations implicate **matmul precision interacting with batch shape**. They do not identify a unique kernel, establish exact arithmetic/bitwise determinism, or prove that gradient maps are now stable. Small MT repeat/order CE differences remain (up to about 1.01e-5 for the focus order control). Focus repeat/order/companion controls apply only to 3896.

## Why stricter precision cannot replace the original baseline

| Model | Original `source_high` correct / 1,631 | `highest_matmul` correct / 1,631 | Predictions differing from source |
|---|---:|---:|---:|
| MT | 1,619 | 1,616 | 3 |
| AMT | 1,613 | 1,612 | 5 |

Both stricter modes have the same changed prediction IDs: MT **3896, 3133, 2870**; AMT **7298, 2870, 7453, 805, 6826**. Only MT recording 3896 is in the 40-record pilot cohort. This is a numerical evaluation deviation, with mixed accuracy/loss effects, not a model improvement or a replacement anchor. Preserve all original high-precision scores and tolerances. No precision winner was automatically selected.

Source-high single-record predictions reproduced all 40 prior pilot predictions for both models. The maximum CE difference from that earlier execution is **1.072883606e-6** for MT and zero for AMT; report this small execution variation rather than claiming exact cross-run equality. Same-run original anchors still pass the original contract. The MT cross-run CE difference slightly exceeds the declared `reproduction_loss_tolerance=1e-6`; the frozen launcher records this drift but does not enforce that field as a gate. Do not claim all cross-run CE reproduction criteria passed. A future gradient protocol must explicitly evaluate and report its own stability criteria.

## Next step

Prepare a **separately versioned gradient follow-up** with fixed `highest_matmul` precision and its own recorded same-mode validation reference. This is a stability-motivated candidate protocol, not a retroactive change to the paper comparison or completed pilot. Require repeat and autograd forward checks, frozen state and input/gradient stability review before expanding pairs/seeds/data. No new gradient maps were generated here, and no full stage-7 completion or causal/architectural superiority is established.

Recorded execute-attempt time: **277.547 seconds (about 4 minutes 38 seconds)**, excluding initial preflight; workflow time is not pure inference latency. This review and its JSON companion are outside the immutable 568-output ledger.

Evidence: [summary](SUMMARY.json), [full validation](FULL_VALIDATION_CHECKS.json), [scores](SCORES.json), [comparisons](COMPARISONS.csv), [state checks](FROZEN_STATE_CHECKS.json), [user tests](CONTRACT_TESTS.txt), [completion](COMPLETE.json), [structured review](POST_RUN_REVIEW.json).
