# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 453640.000000 |
| trainable_parameters | 452900.000000 |
| spikes_total_per_sample | 10553.322880 |
| spikes_l1_per_sample | 8718.452739 |
| spikes_l2_per_sample | 1834.870141 |
| firing_l1_percent | 9.681217 |
| firing_l2_percent | 2.037492 |
| synops_per_sample | 9932026.943463 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.249392 |
| logical_gops_test_padded | 0.249993 |
| forward_ms_per_sample | 1.570755 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
