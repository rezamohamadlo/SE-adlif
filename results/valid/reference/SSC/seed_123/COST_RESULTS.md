# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 1689910.000000 |
| trainable_parameters | 1688435.000000 |
| spikes_total_per_sample | 43681.192523 |
| spikes_l1_per_sample | 18339.695270 |
| spikes_l2_per_sample | 25341.497252 |
| firing_l1_percent | 8.490600 |
| firing_l2_percent | 11.732175 |
| synops_per_sample | 51577850.702335 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.019584 |
| logical_gops_test_padded | 1.020884 |
| forward_ms_per_sample | 1.456407 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
