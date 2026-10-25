# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 451488.000000 |
| trainable_parameters | 450748.000000 |
| spikes_total_per_sample | 10672.888693 |
| spikes_l1_per_sample | 8973.601590 |
| spikes_l2_per_sample | 1699.287102 |
| firing_l1_percent | 9.964542 |
| firing_l2_percent | 1.886937 |
| synops_per_sample | 10062688.604240 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.250372 |
| logical_gops_test_padded | 0.251353 |
| forward_ms_per_sample | 1.793125 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
