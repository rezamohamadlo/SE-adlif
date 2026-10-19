# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4768.000000 |
| trainable_parameters | 4690.000000 |
| spikes_total_per_sample | 26799.439716 |
| spikes_l1_per_sample | 9671.425532 |
| spikes_l2_per_sample | 17128.014184 |
| firing_l1_percent | 20.649555 |
| firing_l2_percent | 36.570190 |
| synops_per_sample | 1415061.531915 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014678 |
| logical_gops_test_padded | 0.015150 |
| forward_ms_per_sample | 43.668066 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
