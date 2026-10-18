# Saved test-set activity and cost

Original evaluator version: `reference_test_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 1689910.000000 |
| trainable_parameters | 1688435.000000 |
| spikes_total_per_sample | 9403.899470 |
| spikes_l1_per_sample | 9029.806839 |
| spikes_l2_per_sample | 374.092631 |
| firing_l1_percent | 4.180466 |
| firing_l2_percent | 0.173191 |
| synops_per_sample | 19366291.558238 |
| dense_gflops_test_padded | 1.008720 |
| arithmetic_gflops_test_padded | 1.019584 |
| logical_gops_test_padded | 1.020884 |
| forward_ms_per_sample | 1.324863 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
