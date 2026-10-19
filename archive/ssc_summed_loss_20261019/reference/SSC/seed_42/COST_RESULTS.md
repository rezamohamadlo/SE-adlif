# Saved test-set activity and cost

Original evaluator version: `reference_test_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 1689910.000000 |
| trainable_parameters | 1688435.000000 |
| spikes_total_per_sample | 8708.322981 |
| spikes_l1_per_sample | 8356.212688 |
| spikes_l2_per_sample | 352.110293 |
| firing_l1_percent | 3.868617 |
| firing_l2_percent | 0.163014 |
| synops_per_sample | 18380568.852909 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.019584 |
| logical_gops_test_padded | 1.020884 |
| forward_ms_per_sample | 1.298403 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
