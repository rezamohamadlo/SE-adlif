# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 0.011053296 |
| tta_dense_gflops | 0.033159888 |
| single_arithmetic_gflops | 0.014677590 |
| tta_arithmetic_gflops | 0.044032770 |
| gain_scaling_gflops | 0.000010408 |
| single_synops_per_sample | 1415061.531914894 |
| tta_synops_per_sample | 4243934.680851064 |
| tta_spikes_per_sample | 80358.085106383 |
| tta_analog_input_dense_macs_per_sample | 562032.000000000 |
| single_forward ms/sample | 39.338105 |
| tta_forwards ms/sample | 115.969230 |
| single_pipeline ms/sample | 39.570615 |
| tta_pipeline ms/sample | 117.629338 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
