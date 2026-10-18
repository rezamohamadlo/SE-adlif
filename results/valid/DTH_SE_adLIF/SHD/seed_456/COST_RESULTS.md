# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 451484.000000 |
| trainable_parameters | 450744.000000 |
| spikes_total_per_sample | 10837.766784 |
| spikes_l1_per_sample | 8289.554770 |
| spikes_l2_per_sample | 2548.212014 |
| firing_l1_percent | 9.204957 |
| firing_l2_percent | 2.829607 |
| synops_per_sample | 9895482.420495 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.250372 |
| logical_gops_test_padded | 0.251353 |
| forward_ms_per_sample | 1.579845 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
