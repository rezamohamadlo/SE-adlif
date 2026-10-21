# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 0.242895231 |
| tta_dense_gflops | 0.728685693 |
| single_arithmetic_gflops | 0.250371899 |
| tta_arithmetic_gflops | 0.751115697 |
| gain_scaling_gflops | 0.000076177 |
| single_synops_per_sample | 9895482.420494700 |
| tta_synops_per_sample | 29683690.477031801 |
| tta_spikes_per_sample | 32507.731890459 |
| tta_analog_input_dense_macs_per_sample | 0.000000000 |
| single_forward ms/sample | 1.809800 |
| tta_forwards ms/sample | 5.396285 |
| single_pipeline ms/sample | 1.748063 |
| tta_pipeline ms/sample | 5.169352 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
