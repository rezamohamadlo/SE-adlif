# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 0.011053296 |
| tta_dense_gflops | 0.033159888 |
| single_arithmetic_gflops | 0.014032626 |
| tta_arithmetic_gflops | 0.042097879 |
| gain_scaling_gflops | 0.000010408 |
| single_synops_per_sample | 1466596.680851064 |
| tta_synops_per_sample | 4400062.340425532 |
| tta_spikes_per_sample | 81082.950354610 |
| tta_analog_input_dense_macs_per_sample | 562032.000000000 |
| single_forward ms/sample | 39.603376 |
| tta_forwards ms/sample | 122.163173 |
| single_pipeline ms/sample | 41.880789 |
| tta_pipeline ms/sample | 127.501217 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
