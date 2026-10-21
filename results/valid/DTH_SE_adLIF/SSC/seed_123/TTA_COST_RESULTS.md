# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 1.008720000 |
| tta_dense_gflops | 3.026160000 |
| single_arithmetic_gflops | 1.025201628 |
| tta_arithmetic_gflops | 3.075604884 |
| gain_scaling_gflops | 0.000084000 |
| single_synops_per_sample | 58326006.622019432 |
| tta_synops_per_sample | 174979516.585712880 |
| tta_spikes_per_sample | 149186.056569522 |
| tta_analog_input_dense_macs_per_sample | 0.000000000 |
| single_forward ms/sample | 2.184931 |
| tta_forwards ms/sample | 6.631859 |
| single_pipeline ms/sample | 2.215092 |
| tta_pipeline ms/sample | 6.693813 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
