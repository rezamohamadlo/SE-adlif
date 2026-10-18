# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 452924.000000 |
| trainable_parameters | 452184.000000 |
| spikes_total_per_sample | 11490.888693 |
| spikes_l1_per_sample | 9069.357332 |
| spikes_l2_per_sample | 2421.531360 |
| firing_l1_percent | 10.070872 |
| firing_l2_percent | 2.688937 |
| synops_per_sample | 10407375.291519 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.251550 |
| logical_gops_test_padded | 0.252338 |
| forward_ms_per_sample | 1.904004 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
