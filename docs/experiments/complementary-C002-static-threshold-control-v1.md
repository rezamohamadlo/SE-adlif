# C002: static versus dynamic thresholds with smoothing

This is a primary-development mechanism control within the
[active plan](complementary-mechanisms-study-v2.md), not a final model claim.
C001 completed its primary and predefined secondary screens. DTH alone failed
the primary comparison. DTH+LS improved over LS by1.53pp on0/3 but only0.40pp
on6/10, below the unchanged0.5pp practical threshold. Keep those outcomes.
Do not tune or extend the secondary pair to rescue them.

The learned DTH models changed both per-neuron static threshold logits and
activity gains. Nonzero gains alone do not establish a benefit from dynamic
traces. Existing SE/EF `train_thr` does not match the bounded per-neuron DTH
parameterization and exact initialization; LT-DTH and FP address other changes.
Learnable thresholds and label smoothing are established ideas. This control
makes no novelty claim.

Compare fresh LS / bounded-static-threshold+LS / DTH+LS. Hold out speakers0/3
using the existing frozen official-training-only indices. All arms use seed42,
batch1024, LR0.01, dropout0.15, float32/high precision, the original300epoch
schedule and a30epoch ceiling. All inherit the dataset's paper baseline;
the intended neuron/objective differences remain explicit. The new static cell
inherits DTH exactly but freezes its two zero activity gains in each layer.
Threshold logits remain trainable. It rejects loading nonzero activity gains.
Retain trace computations for this mechanism comparison; no compute saving is
claimed. No existing checkpoint initializes any arm.

Use clean held-out-speaker accuracy as the primary metric, supported by ordinary
validation CE and curves. The practical threshold remains0.5pp for static-LS
and dynamic-static. If static matches/exceeds dynamic within the declared
screen, dynamic traces are not shown necessary for the observed gain. If dynamic
exceeds static by at least0.5pp with supporting curves/loss, this is conditional
primary-development evidence only. Smaller gaps or loss alone are insufficient.
No automatic transfer, architecture freeze or claim of two complementary ideas
follows. A retained static candidate needs its own ablation and transfer checks.

Do not revisit6/10 for tuning. Keep official tests outside this entire control.
Canonical SHD* and final four-arm comparisons still require fresh training on
their own splits, full budgets, matching seeds and later frozen test evaluation.

Preregistration: `C002_STATIC_THRESHOLD_PREREGISTRATION.json` under
`results/experiments/complementary_speaker_dev_v2/`. New outputs use
`c002_static_threshold_control_v1/primary_0_3/`. Freeze source after correctness
checks and discarded-state calibration; use only the existing runner.

```powershell
.\.ven\Scripts\python.exe runner.py --models LS_SE_adLIF STATIC_LS_SE_adLIF DTH_LS_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_C002_static_control_v1.yaml --stage-ceiling 30 --dry-run
```

The local environment is `.ven`; select the actual environment directory on
another PC. All project artifact paths remain repository-relative. Data and
ignored weights require separate transfer; Git pull does not supply them.
