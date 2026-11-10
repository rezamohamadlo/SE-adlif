# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 452924.000000 |
| trainable_parameters | 452184.000000 |
| spikes_total_per_sample | 12847.044611 |
| spikes_l1_per_sample | 10625.749558 |
| spikes_l2_per_sample | 2221.295053 |
| firing_l1_percent | 11.799134 |
| firing_l2_percent | 2.466589 |
| synops_per_sample | 11449279.010601 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.251550 |
| logical_gops_test_padded | 0.252338 |
| forward_ms_per_sample | 2.067705 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
