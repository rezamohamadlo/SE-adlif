# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 1691354.000000 |
| trainable_parameters | 1689879.000000 |
| spikes_total_per_sample | 41673.948778 |
| spikes_l1_per_sample | 16601.131390 |
| spikes_l2_per_sample | 25072.817388 |
| firing_l1_percent | 7.685709 |
| firing_l2_percent | 11.607786 |
| synops_per_sample | 48878150.975370 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.025202 |
| logical_gops_test_padded | 1.027365 |
| forward_ms_per_sample | 1.704756 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
