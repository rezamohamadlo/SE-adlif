# C001 fixed6/10 pair-sensitivity screen

Fresh baseline and primary-shortlisted LS/DTH+LS; seed42;30/300epochs; no official tests.

| Arm | Online train in selected epoch | Clean validation | Ordinary validation CE | Selected completed epoch | Speaker6 | Speaker10 |
|---|---:|---:|---:|---:|---:|---:|
| SE_adLIF | 98.76% | 85.68% | 0.6125 | 26 | 88.59% | 84.03% |
| LS_SE_adLIF | 99.35% | 87.86% | 0.5879 | 25 | 89.49% | 86.95% |
| DTH_LS_SE_adLIF | 99.22% | 88.27% | 0.5503 | 25 | 89.71% | 87.45% |

- LS_minus_Baseline_pp: +2.184463pp
- combination_minus_Baseline_pp: +2.588993pp
- combination_minus_LS_pp: +0.404531pp

Single predefined secondary pair,one training seed,30/300epochs. Shortlist chosen on primary0/3. Not a pair search, independent population proof, synergy or official-test result. No new tuning/extension on this pair.

Logged online selected-epoch aggregate with dropout and changing weights; not clean checkpoint training reevaluation.

No comparable paper validation metric, so no paper delta. Review original+0.5pp incremental criterion with CE/curves and per-speaker evidence. No automatic next job.
