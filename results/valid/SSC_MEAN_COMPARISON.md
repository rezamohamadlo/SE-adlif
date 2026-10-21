# SSC mean-loss comparison - 21 October 2026

Protocol: ssc_mean_loss_v1; baseline paper_classification_v1, SSC section.

| Model | Seed 42 test (%) | Seed 123 test (%) | Seed 456 test (%) | n | Mean +/- sample SD (%) |
|---|---:|---:|---:|---:|---:|
| Reference | 80.69 | 80.33 | 80.38 | 3 | 80.47 +/- 0.20 |
| DTH | 80.46 | 79.84 | 79.56 | 3 | 79.95 +/- 0.46 |
| MT | 80.01 | 79.98 | 80.23 | 3 | 80.07 +/- 0.14 |

All nine runs completed 40 epochs with batch 256, LR 0.006, mean membrane-potential loss and no early stopping. Checkpoints were selected by validation accuracy. Reference seed 42 retains its diagnostic provenance; the comparison was planned after its test result was observed. Means and sample SD use independent training seeds 42, 123 and 456. No archived summed-loss results are included.

Matching-seed mean test differences from reference, calculated before rounding: DTH -0.512 pp, MT -0.396 pp.

[Full grid](COMPARISON_STATUS.md). [Verified import](../provenance/ssc-local-import-2026-10-21.json).
