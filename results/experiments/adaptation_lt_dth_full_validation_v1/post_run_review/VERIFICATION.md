# LT-DTH/DTH full-validation post-run review

Status: passed. All 63 recorded outputs, source identities, cache identities, full sample coverage, shared inputs/features, pairing, clean anchors and saved summary calculations verified. Six parameter records have matching checkpoint identities, valid trace ordering/bounds and correct frame-to-ms conversion.

Runtime: 345.296 seconds (approximately 5 minutes 45 seconds).

| Clean validation metric | DTH | LT-DTH |
|---|---:|---:|
| Accuracy (mean +/- sample SD, 3 training seeds) | 99.0803 +/- 0.0613% | 99.1416 +/- 0.3186% |

Accuracy SD uses percentage points. The paired mean accuracy gap is +0.0613 pp with sample SD 0.3729 pp. LT-DTH is lower in seeds 42 and 123 and higher in seed 456; this is not a consistent improvement across seeds. These are validation, not final-test, results.

| Model | Seed | Layer | Fast trace e-folding (ms) | Slow trace e-folding (ms) |
|---|---:|---|---:|---:|
| DTH_SE_adLIF | 42 | l1 | 37.965 | 397.997 |
| DTH_SE_adLIF | 42 | l2 | 37.965 | 397.997 |
| DTH_SE_adLIF | 123 | l1 | 37.965 | 397.997 |
| DTH_SE_adLIF | 123 | l2 | 37.965 | 397.997 |
| DTH_SE_adLIF | 456 | l1 | 37.965 | 397.997 |
| DTH_SE_adLIF | 456 | l2 | 37.965 | 397.997 |
| LT_DTH_SE_adLIF | 42 | l1 | 53.310 | 305.926 |
| LT_DTH_SE_adLIF | 42 | l2 | 127.971 | 418.002 |
| LT_DTH_SE_adLIF | 123 | l1 | 51.460 | 292.526 |
| LT_DTH_SE_adLIF | 123 | l2 | 154.179 | 476.233 |
| LT_DTH_SE_adLIF | 456 | l1 | 53.377 | 304.335 |
| LT_DTH_SE_adLIF | 456 | l2 | 161.248 | 629.017 |

These are layer-level decay descriptors using the physical 4-ms input bin. The model has only two parameter records per seed, not 1631 independent learned time constants. Temporal perturbations and feature associations do not by themselves establish a causal explanation.

Next: specify and implement the adjusted LT-DTH/DTH feature-benefit analysis, preserving per-seed estimates and speaker dependence. The AMT/MT statistical specification must not be silently relabeled.

Limitations:
- Descriptive validation-development evidence, not independent final-test confirmation.
- Only three training seeds; clean mean accuracy improvement is not consistent across seeds.
- Learned trace descriptors belong to model/layer units; no per-sample independent parameter observations.
- No new model inference or raw input reconstruction was run by this post-run review.
- LT-DTH-specific adjusted statistical analysis remains pending.
