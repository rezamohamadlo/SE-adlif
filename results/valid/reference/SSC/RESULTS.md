# SSC mean-loss reference

Protocol: ssc_mean_loss_v1. Base: paper_classification_v1, SSC section; mean aggregation is an explicit alternative to historical sum loss. Reference seed 42 reuses the completed diagnostic, retaining its original config/checkpoint bytes and training provenance. More seeds are pending.

| Training seed | Best validation (%) | Selected epoch | Final test (%) | Test loss |
|---|---:|---:|---:|---:|
| 42 | 82.22 | 18 | 80.69 | 0.670582 |

One completed seed; across-seed SD is unavailable. Paper SSC mean is 80.44% across five runs; the +0.25 pp difference is descriptive only. Historical summed-loss seed-42 reference test accuracy was 78.33%; the new result is +2.36 pp, without isolating all execution differences. These are observed results, not proof of the full gap's cause.

[Per-seed result](seed_42/REFERENCE_RESULT.json) and [comparison guide](../../../../docs/experiments/ssc-mean-comparison.md). Measurement costs are pending; they were not inferred from archived summed-loss measurements.

Reporting status (20 October 2026): reference, DTH and MT seed 42 have completed mean-loss SSC training and testing. This reference contributes one accepted matched result to the [21/27 accepted grid](../../COMPARISON_STATUS.md); its remaining seeds are missing locally.
