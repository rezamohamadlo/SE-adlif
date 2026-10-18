# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 450760.000000 |
| trainable_parameters | 450020.000000 |
| spikes_total_per_sample | 9872.035336 |
| spikes_l1_per_sample | 8075.659452 |
| spikes_l2_per_sample | 1796.375883 |
| firing_l1_percent | 8.967441 |
| firing_l2_percent | 1.994747 |
| synops_per_sample | 9454926.810954 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.247824 |
| logical_gops_test_padded | 0.248414 |
| forward_ms_per_sample | 1.253745 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
