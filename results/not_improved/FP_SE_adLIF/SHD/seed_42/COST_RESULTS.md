# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 450760.000000 |
| trainable_parameters | 450020.000000 |
| spikes_total_per_sample | 10097.668286 |
| spikes_l1_per_sample | 8248.235866 |
| spikes_l2_per_sample | 1849.432420 |
| firing_l1_percent | 9.159075 |
| firing_l2_percent | 2.053662 |
| synops_per_sample | 9597163.754417 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.247837 |
| logical_gops_test_padded | 0.248430 |
| forward_ms_per_sample | 1.732852 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
