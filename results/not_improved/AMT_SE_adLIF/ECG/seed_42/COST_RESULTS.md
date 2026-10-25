# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4908.000000 |
| trainable_parameters | 4830.000000 |
| spikes_total_per_sample | 27982.453901 |
| spikes_l1_per_sample | 11078.971631 |
| spikes_l2_per_sample | 16903.482270 |
| firing_l1_percent | 23.654820 |
| firing_l2_percent | 36.090790 |
| synops_per_sample | 1506907.617021 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014879 |
| logical_gops_test_padded | 0.015260 |
| forward_ms_per_sample | 46.877840 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
