# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 1.008720000 |
| tta_dense_gflops | 3.026160000 |
| single_arithmetic_gflops | 1.025201628 |
| tta_arithmetic_gflops | 3.075604884 |
| gain_scaling_gflops | 0.000084000 |
| single_synops_per_sample | 48878150.975370422 |
| tta_synops_per_sample | 146629288.947355509 |
| tta_spikes_per_sample | 125016.098174860 |
| tta_analog_input_dense_macs_per_sample | 0.000000000 |
| single_forward ms/sample | 1.668763 |
| tta_forwards ms/sample | 5.067826 |
| single_pipeline ms/sample | 1.663670 |
| tta_pipeline ms/sample | 5.060786 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
