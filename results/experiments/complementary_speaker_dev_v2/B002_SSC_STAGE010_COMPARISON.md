# B002 LS-only SSC10validation screen

Matched fresh seed42, batch1024, official validation only;10/40epochs.

| Arm | Online train accuracy in selected epoch | Best validation accuracy | Ordinary validation CE | Selected completed epoch |
|---|---:|---:|---:|---:|
| SE_adLIF | 79.48% | 78.43% | 0.7036 | 10 |
| LS_SE_adLIF | 79.91% | 78.81% | 0.7974 | 10 |

LS minus Baseline: +0.380725pp.

Nominal+0.380725pp is below+0.5threshold; LS ordinary validation CE is worse. Both selected epoch10 and improved accuracy/loss between8and10. Train accuracy remains~80%. This is genuinely still-learning/inconclusive within declared40budget, not positive transfer evidence or loss/gap-only success.

Full resume each own last state to original40epoch budget, unchanged optimizer/scheduler/settings/source; preserve10snapshots. No tuning, official tests or extension beyond40.

No official test or comparable published validation baseline; no paper delta. Smoothed training objective is not ordinary validation CE.

Training accuracy is the logged online epoch aggregate with dropout active and weights changing across batches. It is not a clean full-training-set reevaluation of the fixed selected checkpoint. Validation is a clean fixed-checkpoint pass; the gap alone is not a success criterion.
