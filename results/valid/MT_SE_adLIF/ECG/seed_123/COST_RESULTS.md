# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4836.000000 |
| trainable_parameters | 4758.000000 |
| spikes_total_per_sample | 26703.929078 |
| spikes_l1_per_sample | 8643.113475 |
| spikes_l2_per_sample | 18060.815603 |
| firing_l1_percent | 18.453996 |
| firing_l2_percent | 38.561823 |
| synops_per_sample | 1380153.744681 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014033 |
| logical_gops_test_padded | 0.014322 |
| forward_ms_per_sample | 44.521540 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
