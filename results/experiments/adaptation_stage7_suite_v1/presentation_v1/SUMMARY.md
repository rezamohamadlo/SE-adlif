# Stage-7 registered diagnostic scope

Closure: verified_scoped_complete.
All four model pairs, three datasets and three saved training seeds. Frozen weights; no official test.
Full strict validation references are separate from source_high baseline scores.
Primary gradient records: 2520; repeat checks: 432; input-optimization attempts: 216.
Accepted nonzero diagnostic inputs: 171; ineffective/negative attempts remain in full traces.
Shared transfer scores: 5184; paired diagnostic rows: 288.

| Dataset | Pair | Parent-optimized minus random benefit change (mean ± seed SD, CE) |
|---|---|---:|
| SHD* | amt_mt | +1.772487 ± 0.8772572 |
| SHD* | lt_dth | +2.059113 ± 1.193917 |
| SHD* | sa_mt | +0.8798521 ± 1.037291 |
| SHD* | mt_tc | +1.808941 ± 2.903006 |
| SSC | amt_mt | +0.1996852 ± 0.872498 |
| SSC | lt_dth | +1.563769 ± 0.8677277 |
| SSC | sa_mt | +0.9708934 ± 0.6827869 |
| SSC | mt_tc | +1.020111 ± 0.5956798 |
| ECG | amt_mt | -0.0006867465 ± 0.009490185 |
| ECG | lt_dth | +0.02170032 ± 0.04546959 |
| ECG | sa_mt | +0.01428235 ± 0.01235773 |
| ECG | mt_tc | +0.03967725 ± 0.007800522 |

Positive benefit change means the variant loses less supervised performance than its parent under the same diagnostic input.
These are model-selected constrained diagnostic perturbations, not natural accuracy improvements or causal estimates.
Mean/SD summarize three matching training seeds after averaging four fixed records within seed. No p-values or confidence intervals.
See STAGE7_CLOSEOUT_AUDIT.json for scope, checklist and limitations. Stage 8 synthesis and cost reporting remain separate.

Presentation uses explicit SHD* labels and shared diagnostic color scales across six models and three seeds. Frozen scientific outputs remain unchanged.
