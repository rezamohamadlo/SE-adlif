# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 452920.000000 |
| trainable_parameters | 452180.000000 |
| spikes_total_per_sample | 10364.719965 |
| spikes_l1_per_sample | 8676.796820 |
| spikes_l2_per_sample | 1687.923145 |
| firing_l1_percent | 9.634961 |
| firing_l2_percent | 1.874318 |
| synops_per_sample | 9843422.155477 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.250769 |
| logical_gops_test_padded | 0.251555 |
| forward_ms_per_sample | 1.774294 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
