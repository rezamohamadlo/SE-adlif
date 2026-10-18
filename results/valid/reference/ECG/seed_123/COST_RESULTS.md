# Saved test-set activity and cost

Original evaluator version: `reference_test_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 4692.000000 |
| trainable_parameters | 4614.000000 |
| spikes_total_per_sample | 25118.602837 |
| spikes_l1_per_sample | 9580.319149 |
| spikes_l2_per_sample | 15538.283688 |
| firing_l1_percent | 20.455033 |
| firing_l2_percent | 33.175941 |
| synops_per_sample | 1341730.638298 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.013455 |
| logical_gops_test_padded | 0.013740 |
| forward_ms_per_sample | 33.380317 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
