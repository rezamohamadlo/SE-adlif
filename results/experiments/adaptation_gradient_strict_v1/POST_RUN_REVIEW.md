# Fixed-precision gradient pilot: saved-output review

Reviewed 16 November 2026. **The registered pilot's numerical stability checks passed and saved execution is verified. Stage 7 remains incomplete.** The agent performed read-only file/hash/coverage checks and arithmetic audits of saved results; no tests, inference, backward, new gradient summaries, plots or resampling were executed.

Authenticated **456 registered outputs**, **64 scientific source hashes**, both checkpoint file hashes and **17 passed user contracts**. Both same-precision full-validation references passed on **1,631 records** each, with zero prediction mismatches. Aggregate CE drift is 4.6892e-9 (MT) and 7.0095e-9 (AMT). These strict diagnostic references remain separate from original source-high training-validation anchors.

All **164 saved gradients** are finite and nonzero on physical support; all NPZ embedded identities, value digests, array hashes and sidecars agree. All 164 same-mode autograd forward checks and 164 strict-singleton-reference checks passed. The largest absolute autograd CE drift is **2.3246e-6**; the largest strict-reference drift is **2.2650e-6**, within the registered bounds. State/parameter hashes are unchanged and parameter gradients remain absent.

## Stability within the registered scope

All **86 required checks** passed: 80 natural repeat pairs, two padded repeat pairs and four natural-versus-padded comparisons for recording 3896. All **166 JSON/CSV comparison rows** agree, including 80 additional descriptive cross-precision comparisons.

| Comparison | MT max relative L2 difference | AMT max relative L2 difference |
|---|---:|---:|
| Natural repeat | 7.5784e-6 | 1.1275e-5 |
| Focus padded repeat | 1.0115e-6 | 8.6805e-7 |
| Focus natural versus padded | 1.4337e-6 | 2.5087e-6 |
| Strict repeat 0 versus legacy high singleton | 7.7150e-6 | 1.4897e-5 |

All recorded cosines in these groups exceed **0.99999999999**. The required tooling limits are relative L2 <= 0.01 and cosine >= 0.999; these are engineering criteria, not significance tests or proof of correct hard-spike derivatives. These repeat/padding checks do not establish general full-batch gradient invariance. Legacy comparisons concern natural singletons, not the original high-precision full-batch predictions; the earlier three MT / five AMT full-validation precision differences remain valid evidence.

## Recording 3896 remains important

MT still predicts **9 (incorrect)** for this recording in the strict natural singleton, with CE **0.8729861975** and physical absolute-gradient sum **746.7078628793**. AMT predicts **1 (correct)** with CE **9.536697689e-6**. The MT raw-gradient outlier was retained; stability did not remove it or establish an architectural explanation. Do not infer model superiority from raw gradient magnitudes, selected MT-guided inputs or confidence/loss coupling. Normalized per-record profiles have a different weighting from raw-magnitude averages.

## Next step

Prepare a **separate user-launched matched event-deletion pilot** under fixed `highest_matmul`, using predetermined **natural repeat 0** as the primary gradient. Use the same 40 recordings, a 1% exact event budget, MT absolute-gradient guidance on existing event cells, and three exact random controls shared by both models. Preserve all original and strict evidence. Repeat 1 and padded variants remain stability evidence, not new seeds or independent observations. Review input domains, matched budgets and clean references before interpreting perturbation effects. No new training or backward is needed for that preparation.

No deletion evaluations, optimized inputs, expanded seeds/pairs/datasets or official-test results were produced here. Full-validation anchor coverage does not make the gradient pilot full-validation. Whole stage 7 remains open.

Recorded execute-attempt time: **216.313 seconds (about 3 minutes 36 seconds)**, excluding initial preflight; workflow time is not pure inference latency. This review and its JSON companion sit outside the immutable 456-output ledger.

Evidence: [summary](SUMMARY.json), [anchors](FULL_VALIDATION_ANCHORS.json), [checks](STABILITY_CHECKS.json), [check CSV](STABILITY_CHECKS.csv), [gradient index](GRADIENT_INDEX.json), [gradient summaries](GRADIENT_SUMMARIES.json), [state](FROZEN_STATE_CHECKS.json), [user test log](CONTRACT_TESTS.txt), [completion](COMPLETE.json), [structured review](POST_RUN_REVIEW.json).
