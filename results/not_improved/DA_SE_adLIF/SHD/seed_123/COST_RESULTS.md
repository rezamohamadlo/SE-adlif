# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 453640.000000 |
| trainable_parameters | 452900.000000 |
| spikes_total_per_sample | 10565.257951 |
| spikes_l1_per_sample | 8807.904152 |
| spikes_l2_per_sample | 1757.353799 |
| firing_l1_percent | 9.780547 |
| firing_l2_percent | 1.951416 |
| synops_per_sample | 9966336.051237 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.249392 |
| logical_gops_test_padded | 0.249993 |
| forward_ms_per_sample | 1.550618 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
