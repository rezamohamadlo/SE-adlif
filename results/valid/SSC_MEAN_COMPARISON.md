# SSC mean-loss comparison - seed 42

Protocol: ssc_mean_loss_v1. Reference reused from its completed diagnostic; DTH/MT trained fresh. This comparison was planned after observing the reference test result.

| Model | Best validation (%) | Final test (%) | Best epoch | Status |
|---|---:|---:|---:|---|
| reference | 82.22 | 80.69 | 18 | training and testing complete |
| DTH_SE_adLIF | 81.84 | 80.46 | 19 | training and testing complete |
| MT_SE_adLIF | 82.00 | 80.01 | 23 | training and testing complete |

One seed per model; sample SD is unavailable. Seeds 123/456 remain missing locally. DTH is -0.231 pp and MT -0.687 pp relative to the same-seed reference, from unrounded test metrics.

[Full grid](COMPARISON_STATUS.md). No archived summed-loss scores are mixed into these results.
