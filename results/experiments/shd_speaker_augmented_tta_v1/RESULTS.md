# SHD speaker holdout + training augmentation: exploratory test-time augmentation

Same validation-selected checkpoints. Fixed gains 1.00/0.95/1.05; mean class probabilities.
No training or parameter adaptation. Official test results have previously been observed.
TTA uses three model forwards per batch. Ordinary and TTA results remain separate.

| Model | Seed | Selected epoch (zero-based) | Ordinary test (%) | TTA test (%) | Difference (pp) |
|---|---:|---:|---:|---:|---:|
| DTH_SE_adLIF | 42 | 290 | 92.8445 | 92.8445 | +0.0000 |
| DTH_SE_adLIF | 123 | 261 | 93.0654 | 92.9329 | -0.1325 |
| DTH_SE_adLIF | 456 | 140 | 93.5071 | 93.5512 | +0.0442 |
| LG_DTH_SE_adLIF | 42 | 143 | 93.6837 | 93.9929 | +0.3092 |
| LG_DTH_SE_adLIF | 123 | 94 | 96.6431 | 96.7314 | +0.0883 |
| LG_DTH_SE_adLIF | 456 | 249 | 92.4470 | 92.6678 | +0.2208 |

DTH_SE_adLIF, n=3: ordinary mean +/- sample SD 93.1390 +/- 0.3374%; TTA 93.1095 +/- 0.3851%.

LG_DTH_SE_adLIF, n=3: ordinary mean +/- sample SD 94.2580 +/- 2.1562%; TTA 94.4641 +/- 2.0724%.
