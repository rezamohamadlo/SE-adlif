# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 450760.000000 |
| trainable_parameters | 450020.000000 |
| spikes_total_per_sample | 11199.321555 |
| spikes_l1_per_sample | 9227.106007 |
| spikes_l2_per_sample | 1972.215548 |
| firing_l1_percent | 10.246040 |
| firing_l2_percent | 2.190004 |
| synops_per_sample | 10349786.749117 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.247824 |
| logical_gops_test_padded | 0.248414 |
| forward_ms_per_sample | 1.233291 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
