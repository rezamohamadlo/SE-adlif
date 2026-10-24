# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 452920.000000 |
| trainable_parameters | 452180.000000 |
| spikes_total_per_sample | 10361.562279 |
| spikes_l1_per_sample | 8184.950088 |
| spikes_l2_per_sample | 2176.612191 |
| firing_l1_percent | 9.088801 |
| firing_l2_percent | 2.416972 |
| synops_per_sample | 9678219.876325 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.249789 |
| logical_gops_test_padded | 0.250380 |
| forward_ms_per_sample | 1.589336 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
