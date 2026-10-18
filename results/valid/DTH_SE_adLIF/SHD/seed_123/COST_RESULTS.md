# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 451484.000000 |
| trainable_parameters | 450744.000000 |
| spikes_total_per_sample | 9807.354682 |
| spikes_l1_per_sample | 8098.073322 |
| spikes_l2_per_sample | 1709.281360 |
| firing_l1_percent | 8.992330 |
| firing_l2_percent | 1.898035 |
| synops_per_sample | 9437007.500000 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.250372 |
| logical_gops_test_padded | 0.251353 |
| forward_ms_per_sample | 1.595014 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
