# Saved test-set activity and cost

Original evaluator version: `shd_costs_v1`. Original JSON preserved byte-for-byte; see `archive/measurement_provenance/RELOCATION.json` for old paths and hashes.

| Measurement | Value |
|---|---:|
| total_parameters | 452924.000000 |
| trainable_parameters | 452184.000000 |
| spikes_total_per_sample | 10407.871025 |
| spikes_l1_per_sample | 8742.422703 |
| spikes_l2_per_sample | 1665.448322 |
| firing_l1_percent | 9.707834 |
| firing_l2_percent | 1.849361 |
| synops_per_sample | 9883117.111307 |
| dense_gflops_test_padded | 0.242895 |
| arithmetic_gflops_test_padded | 0.251550 |
| logical_gops_test_padded | 0.252338 |
| forward_ms_per_sample | 1.873466 |

Arithmetic/logical totals absent from the original SHD JSON are derived in memory from its recorded scalar-operation breakdown using the existing report convention. No inference was rerun. Activity uses stored sequence lengths; compute/runtime includes batch padding. Logical operations are not hardware FLOPs or energy.
