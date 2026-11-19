# C003: bounded static thresholds and label smoothing, four arms

Follow the [active speaker-development plan](complementary-mechanisms-study-v2.md).
C002's mechanism screen found static+LS87.77%, LS86.73%, dynamic+LS87.03%
on the frozen0/3pair, seed42,30epochs. It did not support a specifically dynamic
benefit. C001's predefined6/10increment was below the0.5pp threshold; that pair
remains closed for tuning. These are development findings, not final/test proof.

Test fresh Baseline / static-only / LS-only / static+LS on the same primary
split. This adds the missing standalone-static contrast and full factorial
interaction. Existing completed cohorts remain separate; no historical weights
or best-looking repeat are substituted. Both components are established ideas;
neither novelty nor synergy is presumed.

Use seed42, batch1024, LR0.01, dropout0.15, float32/high precision, original
300epoch schedule, initially30epochs.75 is conditional on an explicit matched
learning-curve decision. All configs inherit the paper baseline; hidden-cell
and objective deviations are explicit. Both activity gains are frozen at zero
for static cells. Retain trace computation; no inference-speed claim.

The primary metric is clean held-out-speaker accuracy. Assess static-baseline,
combination-static and combination-LS at the original+0.5pp practical threshold,
supported by ordinary validation CE and curves. Report all contrasts and
interaction, including failures. A smaller gap or loss alone is insufficient.
Do not rewrite a failed standalone arm into a successful preregistered result.
This screen does not itself establish two complementary improvements across
datasets; transfer, canonical own-split training, replication and frozen tests
remain required.

## Reproducibility evidence without changing training

Repeated seed42LScontrols differed by1.35pp. Synthetic ordinaryCUDAaggregation
showed small numerical/gradient differences; this does not prove the cause of
the actual score spread. Keep precision and deterministic settings unchanged.
Repeats of one seed are not independent training seeds.

The new opt-in callback records `INITIAL_MODEL_STATE.json` on fresh runs before
the first optimizer update: per-tensor values/dtype/shape hashes, trainability,
CPU/current-GPU RNG hashes and execution flags. It performs no forward pass,
draws no randomness, mutates no state, and refuses to overwrite conflicting
evidence. A full resume skips initial-state recording and retains its own
original evidence. Hashes do not prove identical later numerical trajectories.

Preregistration: `C003_STATIC_FACTORIAL_PREREGISTRATION.json` beside the main
study results. Outputs: `c003_static_factorial_v1/primary_0_3/`. Freeze source
after tests, disposable calibration and runner preflight. Never use calibration
states to initialize a main run. Keep official tests outside development.

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF STATIC_SE_adLIF STATIC_LS_SE_adLIF LS_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_C003_static_factorial_v1.yaml --stage-ceiling 30 --dry-run
```

The local environment is `.ven`; use the actual environment directory on other
PCs. Project paths are repository-relative. Transfer ignored data/checkpoints
separately; Git pull does not supply them.
