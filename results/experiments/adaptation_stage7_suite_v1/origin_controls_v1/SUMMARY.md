# Both predeclared optimization origins

Mean ± sample SD across three matching training seeds after averaging four fixed diagnostic records within seed.
CE contrast = change in parent-minus-variant supervised loss under optimized input minus its matched-random change.
Each origin is selected to increase that model’s own CE; the comparison is conditional on this selection.
Parent-origin positive contrasts and variant-origin negative contrasts must be interpreted together. No natural-accuracy or causal superiority follows.

| Dataset | Pair | Parent-origin CE contrast | Variant-origin CE contrast |
|---|---|---:|---:|
| SHD* | amt_mt | +1.77249 ± 0.877257 | -2.50538 ± 3.18501 |
| SHD* | lt_dth | +2.05911 ± 1.19392 | -2.24387 ± 1.40614 |
| SHD* | sa_mt | +0.879852 ± 1.03729 | -2.53805 ± 1.98844 |
| SHD* | mt_tc | +1.80894 ± 2.90301 | -1.7101 ± 1.89834 |
| SSC | amt_mt | +0.199685 ± 0.872498 | -0.835282 ± 0.968735 |
| SSC | lt_dth | +1.56377 ± 0.867728 | -0.766841 ± 0.952837 |
| SSC | sa_mt | +0.970893 ± 0.682787 | -1.20043 ± 0.358732 |
| SSC | mt_tc | +1.02011 ± 0.59568 | -1.1393 ± 0.812272 |
| ECG | amt_mt | -0.000686747 ± 0.00949018 | -0.0608075 ± 0.0456244 |
| ECG | lt_dth | +0.0217003 ± 0.0454696 | -0.0448117 ± 0.016684 |
| ECG | sa_mt | +0.0142824 ± 0.0123577 | -0.024159 ± 0.0284894 |
| ECG | mt_tc | +0.0396772 ± 0.00780052 | -0.0484189 ± 0.0122057 |
