# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 4772.000000 |
| trainable_parameters | 4694.000000 |
| spikes_total_per_sample | 28529.354610 |
| spikes_l1_per_sample | 9895.326241 |
| spikes_l2_per_sample | 18634.028369 |
| firing_l1_percent | 21.127607 |
| firing_l2_percent | 39.785696 |
| synops_per_sample | 1494379.574468 |
| analog_input_dense_macs_per_sample | 187344.000000 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.014678 |
| logical_gops_test_padded | 0.015151 |
| forward_ms_per_sample | 44.721560 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
