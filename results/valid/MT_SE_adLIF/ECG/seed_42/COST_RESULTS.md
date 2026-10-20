# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4836.000000 |
| trainable_parameters | 4758.000000 |
| spikes_total_per_sample | 27022.510638 |
| spikes_l1_per_sample | 11078.198582 |
| spikes_l2_per_sample | 15944.312057 |
| firing_l1_percent | 23.653170 |
| firing_l2_percent | 34.042856 |
| synops_per_sample | 1466596.680851 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014033 |
| logical_gops_test_padded | 0.014322 |
| forward_ms_per_sample | 45.256762 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
