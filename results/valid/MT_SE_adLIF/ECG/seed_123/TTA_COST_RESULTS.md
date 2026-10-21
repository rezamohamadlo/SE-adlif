# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 0.011053296 |
| tta_dense_gflops | 0.033159888 |
| single_arithmetic_gflops | 0.014032626 |
| tta_arithmetic_gflops | 0.042097879 |
| gain_scaling_gflops | 0.000010408 |
| single_synops_per_sample | 1380153.744680851 |
| tta_synops_per_sample | 4137670.425531914 |
| tta_spikes_per_sample | 80058.581560284 |
| tta_analog_input_dense_macs_per_sample | 562032.000000000 |
| single_forward ms/sample | 67.240053 |
| tta_forwards ms/sample | 194.565574 |
| single_pipeline ms/sample | 66.108847 |
| tta_pipeline ms/sample | 195.916527 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
