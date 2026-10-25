# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4772.000000 |
| trainable_parameters | 4694.000000 |
| spikes_total_per_sample | 29454.843972 |
| spikes_l1_per_sample | 11261.106383 |
| spikes_l2_per_sample | 18193.737589 |
| firing_l1_percent | 24.043698 |
| firing_l2_percent | 38.845626 |
| synops_per_sample | 1574169.148936 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014678 |
| logical_gops_test_padded | 0.015151 |
| forward_ms_per_sample | 46.990685 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
