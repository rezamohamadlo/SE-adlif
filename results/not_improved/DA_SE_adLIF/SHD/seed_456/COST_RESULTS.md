# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 453640.000000 |
| trainable_parameters | 452900.000000 |
| spikes_total_per_sample | 9948.405919 |
| spikes_l1_per_sample | 7930.833481 |
| spikes_l2_per_sample | 2017.572438 |
| firing_l1_percent | 8.806623 |
| firing_l2_percent | 2.240370 |
| synops_per_sample | 9435876.325088 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.249392 |
| logical_gops_test_padded | 0.249993 |
| forward_ms_per_sample | 1.504877 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
