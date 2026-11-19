# Stage 7: independent completion review

**Registered exploratory scope completed.** Executed 16 November 2026; independently reviewed 17 November 2026.
The user explicitly authorized agent experiments through stage-7 completion. All model weights stayed frozen; no training or official-test evaluation occurred.

All four pairs (AMT/MT, LT-DTH/DTH, SA-MT/MT, MT-TC/MT), all three saved training seeds and SHD*/SSC/ECG are covered. Original source-high baselines and historical manifests remain unchanged.

| Verification item | Result |
|---|---:|
| Unique saved checkpoints | 54 |
| Primary gradient records | 2,520 |
| Additional repeat-gradient records/checks | 432 |
| Total independently verified saved gradient arrays | 2,952 |
| Repeat / autograd / optimization-autograd failures | 0 / 0 / 0 |
| Constrained input-optimization attempts | 216 |
| Accepted nonzero diagnostic inputs | 171 |
| Ineffective attempts retained | 45 |
| Shared transfer bundles / target evaluations | 9 / 5,184 |
| Unique shared optimized/random inputs | 864 |
| Paired record/seed/origin rows | 288 |
| Main registered output hashes | 15,439 |
| Common objective/domain/cache tests | 14 passed |

Strict full-reference coverage is 1,631 SHD*, 9,981 SSC and 30 ECG records per checkpoint. Gradient cohorts are **40 SHD*, 70 SSC and all 30 ECG validation records**; only eight records/checkpoint have repeat checks and four have input optimization. Completion of this registered diagnostic scope is not exhaustive SHD*/SSC validation gradients or independent confirmation.

The independent array audit checked cached identities/digests, raw physical gradient summaries including ECG negative events, accepted and rejected candidate traces, native input domains and exact matched budgets. A separate audit checked every transfer score, shared input identity, true-label accuracy count, all 288 paired contrasts and all 24 seed-aggregate groups. All 54 checkpoint files and underlying metrics.csv histories still match their frozen hashes.

All proposals preserve physical length, labels/block masks and per-channel event totals; ECG also preserves positive/negative totals separately. Count-cell caps remain bounded by their original channel maximum. A 1% native-event relocation budget is never accumulated across steps. The three random controls match actual channel/polarity movement allocations. Rejected and zero-change attempts remain recorded.

Accepted attempts: SHD* **47/72**, SSC **69/72**, ECG **55/72**. These counts depend on the task, objective, inputs and budget; they are not a natural-robustness ranking.

## Interpretation requires both input origins

The table shows optimized-minus-matched-random change in parent-minus-variant supervised CE, mean ± sample SD across the three matching training seeds after averaging four fixed records within seed. Random realizations average within record. No p-values or confidence intervals are inferred.

| Dataset | Pair | Parent-optimized origin (CE) | Variant-optimized origin (CE) |
|---|---|---:|---:|
| SHD* | amt_mt | +1.77249 ± 0.877257 | -2.50538 ± 3.18501 |
| SHD* | lt_dth | +2.05911 ± 1.19392 | -2.24387 ± 1.40614 |
| SHD* | sa_mt | +0.879852 ± 1.03729 | -2.53805 ± 1.98844 |
| SHD* | mt_tc | +1.80894 ± 2.90301 | -1.7101 ± 1.89834 |
| SSC | amt_mt | +0.199685 ± 0.872498 | -0.835282 ± 0.968735 |
| SSC | lt_dth | +1.56377 ± 0.867728 | -0.766841 ± 0.952837 |
| SSC | sa_mt | +0.970893 ± 0.682787 | -1.20043 ± 0.358732 |
| SSC | mt_tc | +1.02011 ± 0.59568 | -1.1393 ± 0.812272 |
| ECG | amt_mt | -0.000686747 ± 0.00949018 | -0.0608075 ± 0.0456244 |
| ECG | lt_dth | +0.0217003 ± 0.0454696 | -0.0448117 ± 0.016684 |
| ECG | sa_mt | +0.0142824 ± 0.0123577 | -0.024159 ± 0.0284894 |
| ECG | mt_tc | +0.0396772 ± 0.00780052 | -0.0484189 ± 0.0122057 |

Parent-origin means are positive in 11 of 12 dataset/pair cells; variant-origin means are negative in all 12. This demonstrates strong dependence on the input-selection model. It does not establish that variants universally resist perturbations, explain natural accuracy differences, or causally attend to particular events. Seed effects vary and some reverse direction. Normalized temporal profiles and class maps are surrogate sensitivities with burn-in/initial-condition influences; raw magnitudes remain available.

Held labels do not guarantee unchanged semantic class after event relocation. Six greedy steps do not find a global optimum. ECG uses record-mean CE over temporal targets and model-step time units, with its saved all-zero-vector-to-class-0 policy; no patient IDs or physical milliseconds are inferred.

## Runtime and reproducibility

On the single RTX 4090, the timed execution phase took **3 h 38 min 57 s**. Main figure generation/sealing and internal verification brought the measured queue to **3 h 42 min 21 s**; with startup, plan **about 3 h 45 min** for a fresh run. These are workflow times, not pure inference latency. Initial preflight and this later independent/presentation review are outside the execution-phase timer. Compatible caches resume without repeating completed work.

The exact 112 scientific files and 140 selected native inputs/true targets/masks have portable, independently sealed snapshot ledgers. Source-high scores, checkpoints and prior study packages remain unchanged. The frozen generated main report is retained; the current presentation applies explicit **SHD*** labels and common diagnostic color limits across all six models and three seeds.

Current figures/report: [presentation](presentation_v1/SUMMARY.md), [both input origins](origin_controls_v1/SUMMARY.md), [origin comparison figure](origin_controls_v1/both_optimization_origins.png).
Evidence: [registered closeout](STAGE7_CLOSEOUT_AUDIT.json), [independent array audit](INDEPENDENT_FINAL_VERIFICATION.json), [independent transfer audit](INDEPENDENT_TRANSFER_VERIFICATION.json), [main completion](COMPLETE.json), [raw magnitudes](RAW_GRADIENT_MAGNITUDES.csv), [paired effects](PAIRED_DIAGNOSTIC_EFFECTS.csv), [structured review](POST_RUN_REVIEW.json).

**Next: stage 8 final synthesis and cost/activity reporting.** No stage-8 execution, new model training, independent official-test confirmation or exact Nature analytical-figure reproduction is claimed. This review is separate from the immutable scientific output ledger.
