# B002 primary validation diagnosis: CPU only

Frozen primary speakers0/3, seed42, own validation-selected30epoch checkpoints. Official training file only; no secondary or official-test data. These are diagnostics, not replacement selection scores.

| CPU diagnostic | Baseline | LS0.1 |
|---|---:|---:|
| Overall accuracy | 81.41% | 86.06% |
| Speaker0 accuracy(n=913) | 84.12% | 88.39% |
| Speaker3 accuracy(n=722) | 77.98% | 83.10% |
| Ordinary CE | 0.9171 | 0.5975 |
| ECE15(fraction, lower better) | 0.1191 | 0.1728 |
| Brier(sum over classes, lower better) | 0.3045 | 0.2506 |
| Mean confidence | 0.9310 | 0.6879 |
| Mean wrong-prediction confidence | 0.8105 | 0.4597 |

LS improves CPU accuracy, ordinary CE and Brier and reduces wrong-prediction confidence, but ECE15 worsens: average confidence falls below accuracy. Do not claim uniform calibration improvement or fit a temperature to manufacture an accuracy benefit.

LS corrects133baseline errors but loses57baseline-correct cases;171are wrong under both. CPU net gain is+4.6483pp, positive on each held speaker. No significance or population-independence claim.

171recordings wrong under both selected models; class-specific regressions and persistent confusions remain. These descriptive patterns do not identify a causal mechanism or justify a new heavy combination before transfer checks.

The training-file label dictionary identifies persistent LS confusions:39cases class19(neun)->9(nine),30cases19(neun)->10(null),26cases10(null)->19(neun). These are provisional CPU descriptions, not a validated causal account.

CPU yields7fewer correct baseline and8fewer correct LS recordings versus original GPU score counts. These are net count differences, not measured numbers of changed predictions. At the initial CPU diagnosis the numerical cause was unresolved. The GPU follow-up below reproduces the authoritative scores and explains most differences; no score replacement or precision selection.

Keep original GPU selection scores81.834865%/86.544341%. CPU scores differ by-0.428137/-0.489295pp. Both CPU models use2threads, saved1024validation batches/masks/aggregation, worker0and eval/inference_mode. Confidence is softmax of the existing CE-input aggregated class scores. No fitting, TTA or model-weight changes.

Next: finish matched SSC40, verify ordinary GPU replay on primary validation, then follow the declared ECG transfer gate. Do not extend a flat failure, tune the sensitivity pair, or replace the two-idea/full canonical comparison requirement with these diagnostics.


## GPU replay follow-up

Original high-mode selection correctness counts and CE reproduced; repeated predictions identical. Full-FP32highest mode reduces CPU disagreements from26/30to1/1. This mostly explains the numerical backend discrepancy; it does not justify selecting a different inference mode. High remains authoritative.

| GPU high replay | Baseline | LS0.1 |
|---|---:|---:|
| Overall accuracy | 81.83% | 86.54% |
| Speaker0 accuracy | 84.34% | 89.38% |
| Speaker3 accuracy | 78.67% | 82.96% |
| Ordinary CE | 0.9221 | 0.5944 |
| ECE15 | 0.1145 | 0.1780 |
| Brier | 0.3045 | 0.2500 |

Calibration remains mixed: accuracy/CE/Brier improve, ECE worsens. No scores were replaced, no tuning/TTA or official tests. Exact evidence: `B002_PRIMARY_GPU_REPLAY_VERIFICATION.json`.

Current stage: SSC40closed below its declared practical threshold; matched fresh ECG50screen is running after verified calibration.
