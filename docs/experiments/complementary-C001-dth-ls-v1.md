# C001: DTH and training-only label smoothing

This is a new, separately versioned development hypothesis within the
[active speaker-disjoint plan](complementary-mechanisms-study-v2.md).
The B002 transfer stages are closed; their negative outcomes remain intact.
No final model or successful combination is declared.

Input gain added only 0.18pp over smoothing in the previous primary screen.
Smoothing alone improved held-out SHD speakers, but missed the SSC practical
threshold and regressed in the ECG100 screen. Test a different conditioning
location: existing project DTH changes internal thresholds using previous
per-sample layer activity; smoothing changes the training objective.
Neither fact guarantees complementary generalization.

Adaptive spiking thresholds are established in
[Bellec et al., 2018](https://arxiv.org/abs/1803.09574), and uniform label
smoothing in [Szegedy et al., 2016](https://arxiv.org/abs/1512.00567).
The project DTH uses bounded logistic thresholds with fast/slow layer activity.
This experiment does not claim novelty of either component or their combination.
Historical project tests have already been inspected; eventual official tests
are reused evidence, not pristine independent confirmation. No new test data
enter selection.

## Frozen matched screen

Use fresh Baseline / DTH / DTH+LS / LS, seed42, existing primary0/3 manifest,
batch1024, LR0.01, dropout0.15, ordinary float32/high precision and unchanged
300-epoch scheduler settings. Stop at30 initially;75 is conditional on the
matched learning curves. All configs inherit the paper baseline through the
existing reference and comparison groups. Intended deviations are hidden-cell
type and training-only smoothing epsilon0.1. No event deletion or input filter.
No historical weights or controls are silently substituted.

Compare clean held-out-speaker accuracy, ordinary validation CE and stability.
The practical threshold is +0.5pp for DTH-baseline and combination-DTH; also
require combination-LS >=0.5pp before claiming incremental DTH complementarity.
Report all four contrasts and interaction. A smaller gap alone is insufficient;
combined superiority alone does not prove synergy or significance.
If promising, a separate static-threshold-only control is needed before
attributing a benefit specifically to dynamic traces. Preserve failures.

Only a primary-selected shortlist may enter the predefined6/10 sensitivity
check, with fresh weights. SSC and ECG require later current-model calibration
and their own verified validation protocols. Canonical SHD* requires later
fresh own-split four-arm training. Official tests remain outside discovery.

Preregistration and evidence live under
`results/experiments/complementary_speaker_dev_v2/`, including
`C001_DTH_LS_PREREGISTRATION.json`. Training outputs use
`c001_dth_ls_v1/primary_0_3/`. Freeze source after non-training checks and
disposable throughput confirmation; never initialize study weights from them.
All launches use the existing `runner.py`; no additional runner is introduced.

```powershell
.\.ven\Scripts\python.exe runner.py --models SE_adLIF DTH_SE_adLIF DTH_LS_SE_adLIF LS_SE_adLIF --datasets SHD --seeds 42 --study-file config/studies/complementary_C001_dth_ls_primary_v1.yaml --stage-ceiling 30 --dry-run
```

The local environment is `.ven`; use the checkout's actual environment name on
another PC. Paths are repository-relative; ignored data/checkpoints require
separate transfer. The frozen manifest requires the same official training file.
