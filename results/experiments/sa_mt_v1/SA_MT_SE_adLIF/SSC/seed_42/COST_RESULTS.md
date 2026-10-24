# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 1694230.000000 |
| trainable_parameters | 1692755.000000 |
| spikes_total_per_sample | 47746.356834 |
| spikes_l1_per_sample | 20393.161172 |
| spikes_l2_per_sample | 27353.195663 |
| firing_l1_percent | 9.441278 |
| firing_l2_percent | 12.663517 |
| synops_per_sample | 56040939.204690 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.023918 |
| logical_gops_test_padded | 1.025221 |
| forward_ms_per_sample | 1.750235 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
