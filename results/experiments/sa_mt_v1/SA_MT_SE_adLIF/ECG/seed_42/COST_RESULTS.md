# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4908.000000 |
| trainable_parameters | 4830.000000 |
| spikes_total_per_sample | 27500.397163 |
| spikes_l1_per_sample | 11102.992908 |
| spikes_l2_per_sample | 16397.404255 |
| firing_l1_percent | 23.706108 |
| firing_l2_percent | 35.010258 |
| synops_per_sample | 1487364.000000 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014407 |
| logical_gops_test_padded | 0.014696 |
| forward_ms_per_sample | 42.770970 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
