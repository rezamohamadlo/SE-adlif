# VS-MT sequence: SSC, SHD*, ECG

This extends the [SSC experiment](vs-mt-ssc.md) to SHD* and ECG at the user's
request. The cell equations and initialization are unchanged. Each new config
inherits its dataset's MT architecture and comparison group, including the
corresponding section of `config/paper_baseline.yaml`. SHD settings are never
applied to SSC or ECG.

`runner_vs_mt_all.py` preflights all selected configurations and checks required
local dataset files before starting any job. It then delegates to the official
runner lifecycle in this fixed order:

1. SSC: three seeds, through the existing SSC wrapper and its unchanged root
   `results/experiments/vs_mt_ssc_v1/`.
2. SHD* (`SHD` config identifier): three seeds, under
   `results/experiments/vs_mt_v1/VS_MT_SE_adLIF/SHD/`.
3. ECG: three seeds, under
   `results/experiments/vs_mt_v1/VS_MT_SE_adLIF/ECG/`.

Seeds default to 42, 123 and 456. This is nine training jobs in total; completed
compatible SSC jobs are skipped and interrupted SSC jobs resume from their own
last checkpoint. Later datasets start only after the preceding stage returns
successfully. Config changes and unknown artifacts are rejected by the shared
runner. Source and baseline defaults remain unchanged for existing models.

Run from the main repository root, on a PC with no competing GPU job:

```powershell
.\.venv\Scripts\python.exe runner_vs_mt_all.py --seeds 42 123 456 --dry-run
.\.venv\Scripts\python.exe runner_vs_mt_all.py --seeds 42 123 456
```

Rerun the same command for guarded resume/skip. To select only the later stages:

```powershell
.\.venv\Scripts\python.exe runner_vs_mt_all.py --datasets SHD ECG --seeds 42 123 456 --dry-run
.\.venv\Scripts\python.exe runner_vs_mt_all.py --datasets SHD ECG --seeds 42 123 456
```

Only the user launches training. Do not run this alongside another launcher
targeting either output root. The sequence locks its SHD/ECG root; SSC also
uses the SSC wrapper's original lock. A stale lock file does not mean a lock
is held; never remove one to bypass an active process.

Data must exist locally, preserving these relative paths:

```text
data/SSCWrapper/ssc_train.h5
data/SSCWrapper/ssc_valid.h5
data/SSCWrapper/ssc_test.h5
data/SHDWrapper/shd_train.h5
data/SHDWrapper/shd_test.h5
data/QTDB/QTDB_train.mat
data/QTDB/QTDB_test.mat
```

The SSC comparison reads compatible MT SSC resolved configs and verified
`RESULT.json` records. MT is never a training/resume target. Git transfers
these records but not ignored data, CSV logs, completion flags or weights.
Missing baseline evidence is labeled unavailable; missing baseline weights
only prevent parameter export and do not block VS-MT training. Preflight
verifies dataset presence without loading data or training.
Effective baseline epochs/batch/LR are SSC 40/256/0.006,
SHD* 300/256/0.01, and ECG 400/64/0.01; configs remain authoritative.
SHD* uses 20% training-derived validation with split seed 42. Each dataset's
loss, preprocessing, scheduler and stopping rules remain inherited.

The official runner selects checkpoints on validation, performs final ordinary
test inference without TTA, aggregates completed results and measures new
completions automatically. SSC retains its matching-seed MT comparison.
SHD/ECG use the standard runner result aggregates; their completed VS-MT seed
directories additionally receive `NEURON_PARAMETERS.json` from the selected
checkpoints, recording coefficients, mixtures and time constants.

Use `--measure-existing-costs` to backfill missing costs while skipping completed
training, or `--no-measure-costs` to omit new measurements. Unsupported or failed
costs remain pending and never trigger retraining. Hardware/software differences
limit timing comparisons. Historical tests have already been observed; do not
tune on these test scores or claim guaranteed improvement or novelty.

Non-training checks:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vs_mt_adlif tests.test_vs_mt_ssc_runner tests.test_vs_mt_all
git diff --check
```
