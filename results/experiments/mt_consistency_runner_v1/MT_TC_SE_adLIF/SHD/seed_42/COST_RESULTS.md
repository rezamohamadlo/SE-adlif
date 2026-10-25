# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 452200.000000 |
| trainable_parameters | 451460.000000 |
| spikes_total_per_sample | 11158.321555 |
| spikes_l1_per_sample | 8740.166961 |
| spikes_l2_per_sample | 2418.154594 |
| firing_l1_percent | 9.705329 |
| firing_l2_percent | 2.685187 |
| synops_per_sample | 10168574.646643 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.249006 |
| logical_gops_test_padded | 0.249597 |
| forward_ms_per_sample | 1.631391 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
