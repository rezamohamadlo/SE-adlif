# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 1691358.000000 |
| trainable_parameters | 1689883.000000 |
| spikes_total_per_sample | 45789.560838 |
| spikes_l1_per_sample | 19967.084192 |
| spikes_l2_per_sample | 25822.476646 |
| firing_l1_percent | 9.244020 |
| firing_l2_percent | 11.954850 |
| synops_per_sample | 54278603.073545 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.025202 |
| logical_gops_test_padded | 1.027365 |
| forward_ms_per_sample | 1.898141 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
