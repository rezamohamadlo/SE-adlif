# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 452924.000000 |
| trainable_parameters | 452184.000000 |
| spikes_total_per_sample | 10975.981007 |
| spikes_l1_per_sample | 9252.827297 |
| spikes_l2_per_sample | 1723.153710 |
| firing_l1_percent | 10.274602 |
| firing_l2_percent | 1.913439 |
| synops_per_sample | 10271644.416961 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.251553 |
| logical_gops_test_padded | 0.252536 |
| forward_ms_per_sample | 1.914655 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
