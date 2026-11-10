# Preparation verification — 7 November 2026

State: prepared, no training launched. No active Python training process was
observed during preparation. The six model/seed directories do not exist yet.
The shared root currently contains preparation documentation and the frozen
split manifest only; the launcher creates its OS lock when the user starts it.

The actual official train/test HDF5 files and archive prerequisites are present
locally. Metadata/class coverage and the installed Tonic wrapper's recording
alignment were checked. Split fingerprint:
`e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
The manifest is identical to the archived study's dataset/speaker split identity;
the augmented training protocol and results root are new.

The following six-job preflight passed both from the repository root and from
an external working directory. All six statuses were `new`, in order DTH-42,
LG-DTH-42, DTH-123, LG-DTH-123, DTH-456, LG-DTH-456. It performed no training,
inference, downloads or output writes:

```powershell
.\.venv\Scripts\python.exe runner.py --shd-speaker-phase all --shd-speaker-profile augmented_v1 --dry-run
```

All **35 non-training tests passed**:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_shd_speaker_augmented tests.test_shd_speaker_holdout tests.test_shd_split_cache tests.test_paper_config
```

Checks cover count conservation during time rebinning, non-wrapping channel
shifts, per-spike deletion, reproducible seeded/fresh augmentation draws,
unchanged deterministic caches/validation/test inputs, portable manifest loading
after a working-directory change, all six configurations, changed-setting guards,
full-resume identity/state checks, queue ordering, locking, phase prerequisites,
retirement guards, unchanged baseline composition and restoration of shared-runner
state. No test invokes model training or optimizer steps. These checks do not
establish an accuracy improvement or exercise a full training run.

New local documentation links resolve. The experiment's tracked-source diff and
new files pass whitespace checks. At preparation, the repository-wide
`git diff --check` flagged pre-existing trailing whitespace in
`reports/comparison-summary.md:45`. Before publication, that whitespace was
removed without changing report values; the repository-wide check then passed.
The existing disk-cache reader emitted
`ResourceWarning` messages about unclosed readers in cache tests; tests passed.
`runner.py`, `run.py`, model implementations, `evaluation.yaml` and
`config/paper_baseline.yaml` have no changes from this task.
