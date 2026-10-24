# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4908.000000 |
| trainable_parameters | 4830.000000 |
| spikes_total_per_sample | 27904.539007 |
| spikes_l1_per_sample | 10047.475177 |
| spikes_l2_per_sample | 17857.063830 |
| firing_l1_percent | 21.452462 |
| firing_l2_percent | 38.126791 |
| synops_per_sample | 1472685.191489 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014407 |
| logical_gops_test_padded | 0.014696 |
| forward_ms_per_sample | 43.327647 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
