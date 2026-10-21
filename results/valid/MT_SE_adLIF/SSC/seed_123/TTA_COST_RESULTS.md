# TTA test costs

Frozen validation-selected checkpoint; no training.

| Metric | Per original input sequence |
|---|---:|
| single_dense_gflops | 1.008720000 |
| tta_dense_gflops | 3.026160000 |
| single_arithmetic_gflops | 1.022189993 |
| tta_arithmetic_gflops | 3.066569978 |
| gain_scaling_gflops | 0.000084000 |
| single_synops_per_sample | 58710750.326268278 |
| tta_synops_per_sample | 176119512.688401520 |
| tta_spikes_per_sample | 150154.205818860 |
| tta_analog_input_dense_macs_per_sample | 0.000000000 |
| single_forward ms/sample | 1.811394 |
| tta_forwards ms/sample | 5.406698 |
| single_pipeline ms/sample | 1.838913 |
| tta_pipeline ms/sample | 5.502403 |

GFLOPs cover forwards; gain scaling is separate. Softmax/temporal aggregation/averaging
are excluded from GFLOPs; pipeline timing includes them. Loading/transfer are excluded.
SynOps count event multiplicity with original input counts, not scaled amplitudes;
ECG analog input projection is excluded. No energy measurement is implied.
