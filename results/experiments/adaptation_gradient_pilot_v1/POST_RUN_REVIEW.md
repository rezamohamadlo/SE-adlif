# Stage-7 gradient pilot: saved-output review

Reviewed 16 November 2026. **Pilot execution verified, with a numerical batching follow-up required before expansion. Stage 7 remains incomplete.**

Read-only verification authenticated all **544 registered outputs** and **50 current scientific source hashes**. The user-run log records **18 tests passed**. Both original-batch anchors passed on all **1,631 SHD* validation records** each. All **80 saved input gradients** (MT/AMT, seed 42, 40 fixed records) are finite and nonzero over physical input support. Parameter/buffer hashes are unchanged, with no accumulated parameter gradients. Same-mode forward predictions and CE agree exactly with and without autograd.

The saved 160 deletion arrays have integer, nonpositive deltas, physical-support shapes and exact equal budgets across four controls per recording. Their hashes and shared input identities agree with 400 model evaluation records and 200 paired CSV rows. Saved linear surrogate predictions agree with arithmetic checks against cached gradients/deltas. These checks neither generate perturbations nor execute inference, backward, tests or new statistics. Nonnegative resulting inputs were enforced during the user execution; this review does not reconstruct fresh inputs.

## Numerical batching warning

For **MT recording 3896**, changing from original batching to single-record diagnostic batching changed the prediction from **1 (correct)** to **9 (incorrect)**. CE changed from **0.5342843533** to **0.8729839325**, a difference of **+0.3386995792**. Its raw physical absolute-gradient sum is **746.7067828361**, dominating MT's unnormalized magnitude. The record remains in every applicable saved summary; it is not excluded to improve results. Equal-weight normalized time profiles should be distinguished from raw gradient-magnitude averages.

MT has one source-batch prediction mismatch among 40 pilot records. AMT has none; its largest absolute source-batch CE drift is **0.0004449213**. Within each single-record diagnostic mode, enabling autograd causes **zero** recorded CE drift for both models. The original-batch full-validation anchors passed. The discrepancy therefore concerns the declared batching deviation; its numerical cause has not yet been established. Do not attribute it to precision or a particular kernel without a follow-up.

## Interpretation and next step

This validates the bounded gradient tooling, with the batching warning above. It does not establish model superiority, causal sensitivity or full-validation gradient coverage. The study uses one seed, 40 records, surrogate derivatives and MT-selected deletions with confidence/loss coupling. No input optimization or full stage-7 completion is claimed.

Prepare a **separate user-launched batching/precision sensitivity check** before expanding gradient coverage or interpreting architectural differences. Preserve all pilot outputs, source hashes and original tolerances. Keep recording 3896 and compare its batch-mode behavior explicitly. No training is needed.

Recorded execute-attempt time: **244.985 seconds (about 4 minutes 5 seconds)**, excluding initial preflight; this includes workflow work and is not pure inference latency. The agent only inspected saved evidence. This review and its JSON companion sit outside the frozen 544-output ledger.

Evidence: [summary](SUMMARY.json), [completion](COMPLETE.json), [full anchors](FULL_VALIDATION_ANCHORS.json), [frozen state](FROZEN_STATE_CHECKS.json), [gradient summaries](GRADIENT_SUMMARIES.json), [paired records](PAIRED_RECORDS.csv), [test log](CONTRACT_TESTS.txt), [structured review](POST_RUN_REVIEW.json).
