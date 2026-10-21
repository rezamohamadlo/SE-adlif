# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 0.242895231 |
| tta_dense_gflops | 0.728685693 |
| single_arithmetic_gflops | 0.249005859 |
| tta_arithmetic_gflops | 0.747017576 |
| gain_scaling_gflops | 0.000076177 |
| single_synops_per_sample | 9994329.531802120 |
| tta_synops_per_sample | 29979052.120141342 |
| tta_spikes_per_sample | 32613.000883392 |
| tta_analog_input_dense_macs_per_sample | 0.000000000 |
| single_forward ms/sample | 1.468240 |
| tta_forwards ms/sample | 4.425594 |
| single_pipeline ms/sample | 1.523594 |
| tta_pipeline ms/sample | 4.540957 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
