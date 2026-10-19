# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 1691354.000000 |
| trainable_parameters | 1689879.000000 |
| spikes_total_per_sample | 9091.572368 |
| spikes_l1_per_sample | 8657.488716 |
| spikes_l2_per_sample | 434.083652 |
| firing_l1_percent | 4.008097 |
| firing_l2_percent | 0.200965 |
| synops_per_sample | 18874907.052056 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.025202 |
| logical_gops_test_padded | 1.027365 |
| forward_ms_per_sample | 1.913072 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
