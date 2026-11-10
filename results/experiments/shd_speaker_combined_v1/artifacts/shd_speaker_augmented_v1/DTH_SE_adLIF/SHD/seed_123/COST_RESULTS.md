# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 451484.000000 |
| trainable_parameters | 450744.000000 |
| spikes_total_per_sample | 12519.041961 |
| spikes_l1_per_sample | 10603.939046 |
| spikes_l2_per_sample | 1915.102915 |
| firing_l1_percent | 11.774915 |
| firing_l2_percent | 2.126585 |
| synops_per_sample | 11316532.323322 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.250372 |
| logical_gops_test_padded | 0.251353 |
| forward_ms_per_sample | 1.764670 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
