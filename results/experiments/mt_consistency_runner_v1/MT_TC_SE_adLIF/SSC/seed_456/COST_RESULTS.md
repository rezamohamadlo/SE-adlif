# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 1692790.000000 |
| trainable_parameters | 1691315.000000 |
| spikes_total_per_sample | 50825.767295 |
| spikes_l1_per_sample | 23001.770042 |
| spikes_l2_per_sample | 27823.997252 |
| firing_l1_percent | 10.648968 |
| firing_l2_percent | 12.881480 |
| synops_per_sample | 60142083.342165 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.022190 |
| logical_gops_test_padded | 1.023493 |
| forward_ms_per_sample | 1.810521 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
