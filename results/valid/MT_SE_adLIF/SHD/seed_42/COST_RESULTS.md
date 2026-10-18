# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 452200.000000 |
| trainable_parameters | 451460.000000 |
| spikes_total_per_sample | 11092.889134 |
| spikes_l1_per_sample | 8878.208481 |
| spikes_l2_per_sample | 2214.680654 |
| firing_l1_percent | 9.858615 |
| firing_l2_percent | 2.459244 |
| synops_per_sample | 10189920.945230 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.249006 |
| logical_gops_test_padded | 0.249597 |
| forward_ms_per_sample | 1.637842 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
