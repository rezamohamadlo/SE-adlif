# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 450760.000000 |
| trainable_parameters | 450020.000000 |
| spikes_total_per_sample | 10260.705830 |
| spikes_l1_per_sample | 8163.590989 |
| spikes_l2_per_sample | 2097.114841 |
| firing_l1_percent | 9.065083 |
| firing_l2_percent | 2.328696 |
| synops_per_sample | 9632096.307420 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.247724 |
| logical_gops_test_padded | 0.248314 |
| forward_ms_per_sample | 1.212646 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
