# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 1694230.000000 |
| trainable_parameters | 1692755.000000 |
| spikes_total_per_sample | 47397.897900 |
| spikes_l1_per_sample | 20394.422579 |
| spikes_l2_per_sample | 27003.475321 |
| firing_l1_percent | 9.441862 |
| firing_l2_percent | 12.501609 |
| synops_per_sample | 55782393.287705 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.026077 |
| logical_gops_test_padded | 1.027811 |
| forward_ms_per_sample | 2.146962 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
