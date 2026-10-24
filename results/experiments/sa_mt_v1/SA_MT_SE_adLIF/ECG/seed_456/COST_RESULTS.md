# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4908.000000 |
| trainable_parameters | 4830.000000 |
| spikes_total_per_sample | 26094.631206 |
| spikes_l1_per_sample | 10202.851064 |
| spikes_l2_per_sample | 15891.780142 |
| firing_l1_percent | 21.784207 |
| firing_l2_percent | 33.930695 |
| synops_per_sample | 1401398.510638 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014407 |
| logical_gops_test_padded | 0.014696 |
| forward_ms_per_sample | 41.178998 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
