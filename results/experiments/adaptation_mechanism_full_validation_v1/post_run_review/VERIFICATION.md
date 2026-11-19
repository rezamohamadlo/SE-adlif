# Full-validation post-run review

Status: passed. 57 recorded output hashes, source identities, 36 prediction chunks, six shared feature chunks, six clean anchors, full split membership, shared feature/input identity and parent/variant pairing verified.

All 1,631 validation samples are present for each of three training seeds and six conditions/replicates. There are 29,358 unique paired records; repeated display of clean predictions across reports is not new evidence.

| Clean validation metric | MT | AMT |
|---|---:|---:|
| Accuracy (mean +/- sample SD across 3 training seeds) | 99.2643 +/- 0.0613% | 99.0190 +/- 0.1062% |
| Supervised loss (mean +/- sample SD) | 0.06651386 +/- 0.02405521 | 0.07385750 +/- 0.02262506 |

Accuracy SD uses percentage points. These are validation results, not final-test accuracy.

| Perturbation | Change in loss benefit from clean (mean +/- sample SD) | AMT - MT accuracy gap (pp, mean +/- sample SD) |
|---|---:|---:|
| time_scale_0.9 | -0.00872799 +/- 0.01145030 | -0.2452 +/- 0.0613 |
| time_scale_1.1 | +0.00211418 +/- 0.02836802 | -0.3883 +/- 0.7788 |
| delete_0.1 | +0.00147615 +/- 0.02225297 | -0.3611 +/- 0.4298 |

Benefit is loss(MT) - loss(AMT); positive favors AMT. Deletion realizations are averaged inside each training seed first. Sample SD is not a confidence interval.

Evaluation runtime from COMPLETE.json: 351.734 seconds (approximately 5 minutes 52 seconds).

AMT has lower clean validation accuracy than MT in each of the three seeds. The perturbation effects vary by seed; these descriptive estimates do not establish a consistent AMT advantage. Negative and inconclusive findings are retained.

Next: prepare the Stage 5 analysis specification for feature-benefit relationships, controlling class, valid duration and overall activity and accounting for repeated samples, training seeds and speakers. Freeze effect definitions, uncertainty estimation and multiplicity rules before interpreting confirmatory claims. Do not use official-test results to choose a feature or narrative.

Limitations:
- Validation-development evidence, not independent official-test confirmation.
- Descriptive effects and sample SD across only three training seeds; controlled Stage 5 inference remains pending.
- Full-review checks event totals and duration in saved feature records; it does not rerun model inference or independently reconstruct raw transformed inputs.
- Clean records appear in multiple analysis reports but are shared predictions, not independent observations.
