# Test-set activity and forward costs

Validation-selected checkpoint; no training. See `COST_RESULTS.json` for provenance and exact counts.

| Measurement | Value |
|---|---:|
| total_parameters | 452924.000000 |
| trainable_parameters | 452184.000000 |
| spikes_total_per_sample | 10595.747350 |
| spikes_l1_per_sample | 8311.162986 |
| spikes_l2_per_sample | 2284.584364 |
| firing_l1_percent | 9.228951 |
| firing_l2_percent | 2.536867 |
| synops_per_sample | 9810441.563604 |
| analog_input_dense_macs_per_sample | 0.000000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.251553 |
| logical_gops_test_padded | 0.252536 |
| forward_ms_per_sample | 2.120955 |

Activity is per stored sequence (including minimum padding/burn-in); compute includes batch padding.
ECG SynOps exclude analog input; arithmetic/logical estimates are not exact full end-to-end FLOPs or energy.
