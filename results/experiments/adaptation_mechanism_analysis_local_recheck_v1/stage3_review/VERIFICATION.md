# Stage 3 evidence verification

Saved evidence verification passed. No training or inference was run by this review.

Original clean metadata: `archive/pull-backups/20261112-073929/results/experiments/adaptation_mechanism_analysis_v1/pc1`. Detailed clean rows remain in `results/experiments/adaptation_mechanism_analysis_v1/pc1/diagnostic_rows`. Keep these together; do not pair them with imported completion records.

| Scope | Seed | Samples | MT accuracy (%) | AMT accuracy (%) | MT loss | AMT loss | Benefit (MT - AMT loss) |
|---|---:|---:|---:|---:|---:|---:|---:|
| pilot_256 | 42 | 256 | 99.2188 | 99.6094 | 0.049249303 | 0.056841289 | -0.007591986 |
| full_clean_1631 | 42 | 1631 | 99.2643 | 98.8964 | 0.093870496 | 0.072373059 | +0.021497437 |
| pilot_256 | 123 | 256 | 99.6094 | 100.0000 | 0.050395188 | 0.003242118 | +0.047153070 |
| full_clean_1631 | 123 | 1631 | 99.2029 | 99.0803 | 0.057002440 | 0.052011207 | +0.004991234 |
| pilot_256 | 456 | 256 | 99.6094 | 98.8281 | 0.026927471 | 0.055336205 | -0.028408734 |
| full_clean_1631 | 456 | 1631 | 99.3256 | 99.0803 | 0.048668635 | 0.097188211 | -0.048519576 |

Means and sample SD across three distinct training seeds (equal weight per seed):

- pilot_256: parent_accuracy: 99.479166667 +/- 0.225527449; variant_accuracy: 99.479166667 +/- 0.596689544; parent_loss: 0.042190654 +/- 0.013230715; variant_loss: 0.038473204 +/- 0.030520295; benefit: 0.003717450 +/- 0.039029784. Accuracy units: percent; accuracy SD: percentage points.
- full_clean_1631: parent_accuracy: 99.264255058 +/- 0.061312078; variant_accuracy: 99.019006744 +/- 0.106195635; parent_loss: 0.066513857 +/- 0.024055193; variant_loss: 0.073857492 +/- 0.022625054; benefit: -0.007343635 +/- 0.036602012. Accuracy units: percent; accuracy SD: percentage points.

Verified all listed output hashes for:
- original_clean: 19 files.
- local_timing: 31 files.
- local_deletion: 37 files.

Six clean anchors were recomputed from saved rows and matched saved metrics within study tolerances. Source/checkpoint/metrics hashes, validation selection, full sample membership and parent/variant pairing passed.

Limitations:
- Original clean metadata remains in an ignored local backup; retain or transfer it with the mapped detailed files.
- Imported remote completion records are separate and were not substituted for local ones.
- This review verifies saved evidence; it is not a new inference or independent test.
- Perturbations remain a 256-sample pilot; only clean anchors cover all 1631 validation samples.
