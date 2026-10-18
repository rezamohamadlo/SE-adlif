# Saved test-set activity and cost

Original evaluator version: `reference_test_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 4692.000000 |
| trainable_parameters | 4614.000000 |
| spikes_total_per_sample | 28303.524823 |
| spikes_l1_per_sample | 11726.049645 |
| spikes_l2_per_sample | 16577.475177 |
| firing_l1_percent | 25.036403 |
| firing_l2_percent | 35.394729 |
| synops_per_sample | 1539726.553191 |
| dense_gflops_test_padded | 0.011053 |
| arithmetic_gflops_test_padded | 0.013455 |
| logical_gops_test_padded | 0.013740 |
| forward_ms_per_sample | 38.640362 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
