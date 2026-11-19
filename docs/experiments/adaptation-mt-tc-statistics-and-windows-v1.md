# MT-TC statistics and temporal-evidence diagnostics

User execution completed 12 November 2026 for both paths. Read-only checks
matched the completion identities and all eight statistics and ten window
diagnostic output hashes. See the [combined saved-result review](../../results/experiments/adaptation_mt_tc_full_validation_v1/MT_TC_ANALYSIS_REVIEW.md).
No rerun is required. Tests have not been run by the agent, and no analysis or
inference was rerun during inspection. Original evaluation records remain unchanged.

These are two different analyses. Do not treat the input-feature regression as
a direct test of temporal class-evidence stability, or manufacture window
predictions from aggregate losses. The existing evaluation saved no window
outputs, so the diagnostic needs fresh frozen-model inference, launched by the
user. Neither command trains, tunes checkpoints, uses TTA, downloads data or
opens the official test set.

## 1. Saved-result adjusted statistics

[Specification](../../config/studies/adaptation_mt_tc_statistics_v1.yaml):
the completed MT-TC/MT source manifest and completion hashes are pinned.
The response is supervised `loss(MT) - loss(MT-TC)`, excluding auxiliary JS.
Fast/slow input contrast and input ACF are both two-sided exploratory
associations here. Controls, fixed scaling, speaker-cluster resampling and
multiplicity families inherit the established [statistical method](adaptation-statistics-v1.md).
Three training seeds and ten speakers limit inference. Reused MT predictions
do not create independent runs; percentile intervals remain approximate and
conditional on these model pairs. The specification is after descriptive
validation inspection and is not preregistration.

```powershell
# Optional preflight; no regressions/bootstrap/result writes.
.\.venv\Scripts\python.exe scripts/adaptation_tc_statistics/analyze.py --check-only

# USER LAUNCH: saved-prediction statistics, CPU only.
.\.venv\Scripts\python.exe scripts/adaptation_tc_statistics/analyze.py
```

Outputs are under the completed MT-TC result directory's `statistics_v1/`.
They include manifest, summary, corrected approximate intervals, exclusions,
sample CSV, bootstrap arrays, figures and completion hashes. No direct
output-window evidence claim is made by this analysis.

## 2. Direct clean-input output-window diagnostic

[Specification](../../config/studies/adaptation_mt_tc_windows_v1.yaml): use the
same 1,631 validation samples and six selected MT/MT-TC checkpoints. Conditions
are **clean only**. Both models require new clean forward passes because the
saved parent predictions also lack temporal outputs. The model inputs, reset,
minimum padding and ordinary prediction loss remain identical to the completed
evaluation. Ordinary clean predictions must match exactly; mean loss drift
must remain within the established tolerance. Per-sample loss drift is recorded.
If that gate fails, stop before interpreting window metrics.

Descriptor masks differ deliberately from the ordinary supervised-loss mask.
Exclude the first ten burn-in outputs and all minimum/batch padding beyond
the original physical frame support. Burn-in inputs still affect model state;
this does not remove them from the forward pass. Partition the remaining
support into **nonoverlapping five-frame windows, stride five** (20 ms with
the saved four-ms bin). Drop and record an incomplete tail. At least two
complete windows are required; otherwise retain an explicit undefined reason.
These windows differ from training JS's overlapping window/stride-one scheme.

Convert each frame's output to class softmax probabilities and average those
probabilities within each window. Compute:

- Mean adjacent-window Jensen–Shannon divergence in nats: a descriptive
  disagreement measure. Lower is smoother, not necessarily more useful/correct.
- Mean true-class probability and fraction of windows whose argmax matches
  the label: class alignment checks alongside smoothness.
- Fraction of adjacent windows whose predicted class changes.
- True-class evidence concentration: normalize each window's true-class mass
  by its sum over complete windows, then compute `(K * sum(q**2) - 1)/(K - 1)`.
  Zero indicates uniform temporal mass and one indicates mass in a single
  window. Zero total true-class mass is explicitly undefined for concentration.

Save sample IDs, labels, speakers, input hashes, checkpoint/model/seed identity,
support/window counts, quality reasons, true-class window probabilities and
window predicted classes. These quantities do not recover within-window timing
or prove that a brief event caused the record-level decision. Stable low or
wrong class evidence must not be interpreted as successful temporal reasoning.

Average scalar descriptors **per sample**, then report each model/seed and
paired MT-TC-minus-MT differences. Across-seed means/sample SD use three seed
values; windows are never pooled as independent runs. The diagnostic is
descriptive, with no p-values or automatic benefit regression. Regressing
benefit on parent-derived confidence/stability can introduce mathematical
coupling and post-treatment interpretation issues; such inference would need
a separate design, not the existing input-feature model.

```powershell
# Optional preflight: authenticated source/data/checkpoints, no forward passes.
.\.venv\Scripts\python.exe scripts/adaptation_tc_windows/evaluate.py --check-only

# USER LAUNCH: GPU frozen-model window inference; no training.
.\.venv\Scripts\python.exe scripts/adaptation_tc_windows/evaluate.py
```

Outputs are under `results/experiments/adaptation_mt_tc_full_validation_v1/window_evidence_v1/`:
manifest, six per-model/seed chunks in `diagnostic_rows/`, clean compatibility
anchors, summaries, progress and completion hashes. Whole-model chunks are
identity/digest checked on continuation. No modified inputs are averaged as TTA.
This is a short local workflow; do not start concurrent writers for the same
directory. Progress remains visible in the current terminal and `PROGRESS.json`.

## Verify and share

After USER execution:

```powershell
.\.venv\Scripts\python.exe scripts/adaptation_tc_statistics/analyze.py --verify-only
.\.venv\Scripts\python.exe scripts/adaptation_tc_windows/evaluate.py --verify-only
```

Transfer each COMPLETE directory, including ignored `diagnostic_rows/` (CSV,
bootstrap arrays or diagnostic chunks). Completed-package verification does
not need checkpoints, datasets or a GPU. Execution/continuation still requires
the original frozen evaluation package and local data/weights. Paths resolve
from the checkout root, independently of Hydra/current working directory.
Changed code/settings/environment require a new output root; guards must not
be removed or rewritten. The original comparison records are preserved.

User-run tests, not run by the agent:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_tc_windows tests.test_adaptation_statistics
```

After both runs, inspect per-seed effects and model evidence jointly, preserve
negative/inconclusive findings, and update the roadmap. Neither this diagnostic
nor the feature regression establishes causality or independent final-test
confirmation. Later datasets, saliency or training are not started automatically.
