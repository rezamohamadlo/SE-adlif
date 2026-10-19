# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4836.000000 |
| trainable_parameters | 4758.000000 |
| spikes_total_per_sample | 26269.503546 |
| spikes_l1_per_sample | 10114.226950 |
| spikes_l2_per_sample | 16155.276596 |
| firing_l1_percent | 21.594985 |
| firing_l2_percent | 34.493288 |
| synops_per_sample | 1406039.234043 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014033 |
| logical_gops_test_padded | 0.014322 |
| forward_ms_per_sample | 44.230404 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
