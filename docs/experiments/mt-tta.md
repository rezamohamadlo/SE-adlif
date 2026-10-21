# MT test-time augmentation

DTH is also supported using `--models DTH_SE_adLIF --tta`, with the identical gain policy and separate default outputs under `results/experiments/dth_tta_v1/`. The same validation-first workflow applies. All three datasets now have local checkpoints for seeds 42/123/456. No model training is invoked by these commands.

This is frozen-checkpoint inference through the official `runner.py`, not training or test-time adaptation. It covers SHD*, SSC and ECG, with seeds 42/123/456 when their selected checkpoints are available. It never updates parameters, optimizers, original results or checkpoints.

## Fixed exploratory policy

`config/tta.yaml` specifies three views per dataset: identity and input gains 0.95/1.05. Each view starts from the model's normal reset state. Predictions use the checkpoint's original temporal aggregation; class probabilities are averaged across views. Identity-only inference is measured on the exact same batches for comparison.

Timing, padding masks and targets are unchanged. For ECG, outputs remain aligned to their per-timestep labels. For SHD*/SSC, gains scale preprocessed event-count features, creating fractional intensity views; this is not a raw-event augmentation. For ECG, gains scale the existing preprocessed input features, not the original analog recording. Semantic label invariance is an experimental assumption; small gains are not guaranteed to improve accuracy. No time shifts, reversal or label remapping are used.

Use validation results to decide whether this fixed policy is useful before requesting official test evaluation. Do not search transformation strengths using test accuracy. These datasets' historical test scores have already been observed, so later TTA testing is exploratory rather than a previously untouched independent evaluation. Three forward passes also cost more than the original single-view evaluation; no speed or energy advantage is claimed.

## Commands

Check all nine desired runs without inference or training:

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --tta --dry-run
```

Current local availability (21 October 2026): all nine MT checkpoints and all nine DTH checkpoints, including imported SSC seeds 123/456 with configs and original metrics. Missing selections prevent evaluation from starting; the dry run lists them.

Evaluate all nine available checkpoints on validation:

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SHD SSC ECG --seeds 42 123 456 --tta
```

The all-dataset command without `--dry-run` evaluates validation for all nine. After freezing the policy, adding `--tta-split test` explicitly evaluates the official test split. Do not use training/resume overrides with `--tta`.

Outputs are separate under `results/experiments/mt_tta_v1/<dataset>/seed_<seed>/<validation-or-test>/`. Each contains a manifest with checkpoint SHA256, resolved config, policy, source hashes and environment, plus a result comparing identity and TTA on the same data. The root `RESULTS.md` summarizes per-seed values and completed-seed mean/sample SD. Existing matching completions are skipped; policy/config/source changes require a new `--tta-output` directory. Results stay separate from unaugmented comparison tables.

Use the project `.venv` and preserve repository-relative artifact paths when transferring computers. No training or full-dataset inference was launched during preparation; only synthetic tests and checkpoint/data preflight were performed.


## MT-DTH evaluation

The same frozen input-gain policy supports `MT_DTH_SE_adLIF`. Its existing SHD* checkpoints are resolved from `results/not_improved/MT_DTH_SE_adLIF/SHD/`; no SSC/ECG checkpoints currently exist. Outputs use a separate `results/experiments/mt_dth_tta_v1/` directory.

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_DTH_SE_adLIF --datasets SHD --seeds 42 123 456 --tta --tta-split test
```

This performs test evaluation only, without training. The checkpoints and earlier test results have already been inspected; retain the exploratory designation. Omit `--tta-split test` to evaluate validation instead. Changes to evaluator source preserve manifest guards: use a new output directory when revisiting older MT/DTH evaluations created with different source hashes.
