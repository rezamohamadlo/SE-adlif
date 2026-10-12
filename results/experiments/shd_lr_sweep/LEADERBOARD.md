# SHD validation sweep

Sorted by acc_diffe descending; untested trials appear last. Ties use validation accuracy, then validation loss. BEST_CONFIG.json retains the best validation configuration.
`acc_diffe` is test accuracy minus the reported reference test accuracy (93.42%), in percentage points.
Delta is relative to the same saved reference weights evaluated before each trial.

| Trial | LR | Batch | Scheduler | Val accuracy | Delta (pp) | Val loss | Test accuracy | acc_diffe (pp) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| c03 | 0.0001 | 512 | plateau | 98.958% | -0.184 | 0.07900 | 93.68% | +0.26 |
| c19 | 0.01 | 512 | plateau | 98.774% | -0.368 | 0.07384 | 93.68% | +0.26 |
| c13 | 0.003 | 512 | constant | 98.712% | -0.429 | 0.08961 | 93.68% | +0.26 |
| c07 | 0.0003 | 512 | plateau | 98.958% | -0.184 | 0.07979 | 93.60% | +0.18 |
| c04 | 0.0001 | 512 | cosine | 99.080% | -0.061 | 0.07994 | 93.51% | +0.09 |
| c06 | 0.0003 | 256 | constant | 99.080% | -0.061 | 0.08665 | 93.51% | +0.09 |
| c08 | 0.0003 | 512 | cosine | 99.019% | -0.123 | 0.08368 | 93.46% | +0.04 |
| c01 | 0.0001 | 512 | constant | 99.142% | +0.000 | 0.08031 | 93.37% | -0.05 |
| c20 | 0.01 | 512 | cosine | 98.712% | -0.429 | 0.08562 | 93.33% | -0.09 |
| c11 | 0.001 | 512 | plateau | 98.896% | -0.245 | 0.09632 | 93.29% | -0.13 |
| c05 | 0.0003 | 512 | constant | 98.835% | -0.307 | 0.08575 | 93.29% | -0.13 |
| c02 | 0.0001 | 256 | constant | 98.958% | -0.184 | 0.08740 | 93.20% | -0.22 |
| c16 | 0.003 | 512 | cosine | 98.835% | -0.307 | 0.10522 | 93.15% | -0.27 |
| c10 | 0.001 | 256 | constant | 98.896% | -0.245 | 0.09191 | 93.02% | -0.40 |
| c15 | 0.003 | 512 | plateau | 98.590% | -0.552 | 0.09861 | 92.98% | -0.44 |
| c12 | 0.001 | 512 | cosine | 99.080% | -0.061 | 0.07882 | 92.76% | -0.66 |
| c17 | 0.01 | 512 | constant | 98.222% | -0.920 | 0.08123 | 92.36% | -1.06 |
| c14 | 0.003 | 256 | constant | 98.896% | -0.245 | 0.07515 | 92.27% | -1.15 |
| c09 | 0.001 | 512 | constant | 98.896% | -0.245 | 0.09180 | 92.23% | -1.19 |
| c18 | 0.01 | 256 | constant | 98.406% | -0.736 | 0.09920 | 91.03% | -2.39 |

20/20 trials completed.
Short continuation ranks fine-tuning settings; it does not establish the best settings for training from scratch.
