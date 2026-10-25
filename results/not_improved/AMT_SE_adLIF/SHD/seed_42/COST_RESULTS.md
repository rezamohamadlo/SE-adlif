# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 452920.000000 |
| trainable_parameters | 452180.000000 |
| spikes_total_per_sample | 11115.968640 |
| spikes_l1_per_sample | 8543.742933 |
| spikes_l2_per_sample | 2572.225707 |
| firing_l1_percent | 9.487215 |
| firing_l2_percent | 2.856272 |
| synops_per_sample | 10086493.966431 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.250769 |
| logical_gops_test_padded | 0.251555 |
| forward_ms_per_sample | 1.984108 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
