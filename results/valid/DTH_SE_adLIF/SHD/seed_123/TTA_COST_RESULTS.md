# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 0.242895231 |
| tta_dense_gflops | 0.728685693 |
| single_arithmetic_gflops | 0.250371899 |
| tta_arithmetic_gflops | 0.751115697 |
| gain_scaling_gflops | 0.000076177 |
| single_synops_per_sample | 9437007.500000000 |
| tta_synops_per_sample | 28309200.053003535 |
| tta_spikes_per_sample | 29419.445229682 |
| tta_analog_input_dense_macs_per_sample | 0.000000000 |
| single_forward ms/sample | 1.612621 |
| tta_forwards ms/sample | 4.938983 |
| single_pipeline ms/sample | 1.638049 |
| tta_pipeline ms/sample | 4.942995 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
