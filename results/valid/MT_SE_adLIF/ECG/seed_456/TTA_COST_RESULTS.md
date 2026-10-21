# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 0.011053296 |
| tta_dense_gflops | 0.033159888 |
| single_arithmetic_gflops | 0.014032626 |
| tta_arithmetic_gflops | 0.042097879 |
| gain_scaling_gflops | 0.000010408 |
| single_synops_per_sample | 1406039.234042553 |
| tta_synops_per_sample | 4218208.723404256 |
| tta_spikes_per_sample | 78823.390070922 |
| tta_analog_input_dense_macs_per_sample | 562032.000000000 |
| single_forward ms/sample | 67.526703 |
| tta_forwards ms/sample | 197.466623 |
| single_pipeline ms/sample | 66.214819 |
| tta_pipeline ms/sample | 199.297931 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
