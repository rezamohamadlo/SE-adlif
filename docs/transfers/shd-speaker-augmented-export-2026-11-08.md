# SHD speaker-holdout results export, 8 November 2026

The user requested a Git push of the ordinary/TTA results and a local transfer
bundle at external storage `G:/reza`. The original checkout and all original
artifacts are retained. No training is started, stopped or resumed by this export.

## Bundle layout

The bundle preserves repository-relative paths. Copy these two folders into
the main PC checkout's `results/experiments/` directory:

- `results/experiments/shd_speaker_augmented_v1/`
- `results/experiments/shd_speaker_augmented_tta_v1/`

The first includes DTH and LG-DTH seeds 42/123/456: all six best checkpoints,
all six last checkpoints, original metrics, completion markers, runtime logs,
Hydra configs, split manifests, resolved configs, summaries and cost evidence.
The second includes the six TTA result/manifest pairs, the evaluation recipe
and aggregate summary. Ordinary and exploratory TTA evidence remain separate.

The SHA-256 inventory is
`results/provenance/shd-speaker-augmented-export-2026-11-08.json`.
This guide is included under the same `docs/transfers/` path in the bundle.
The inventory records every payload file's repository-relative path, byte size
and hash. It excludes itself to avoid a circular hash. It is copied and checked
separately. No existing different destination file is overwritten during export.

Lock files and temporary/cache files are excluded. Model weights, raw metrics,
completion markers and logs remain excluded from Git, but are in the local
bundle. The Git push retains configs, summaries, measurement records, TTA
results, the recipe and transfer provenance. No dataset or environment is
copied: the main PC's existing SHD files must match the frozen split manifest.

## On the main PC

Merge the two experiment folders from the bundle into the same relative
locations in that checkout, retaining their complete seed-directory layout.
If the regularization queue is active, import these artifacts first and defer
`git pull` until that queue finishes: changing the checkout's Git revision can
invalidate the recorded source guard needed for an interrupted run's resume.
The result import itself does not change the code revision. Check any differing
existing file before replacing it; a different checkpoint is not a substitute.
Finish copying each run's evidence before exposing its `completed.txt` marker
to a running comparison. Copy completion markers last when transferring in
separate steps. Do not transfer lock files from either machine.

The new regularization queue can continue while the baseline is absent. Once
the completed baseline evidence is present, its next completed candidate
refreshes the validation comparison. If that queue has already completed,
rerunning its original command skips verified completed training and refreshes
the comparison. An already running queue does not need to be restarted.

To check the bundle hashes in PowerShell before importing (adjust the external
drive letter if needed):

```powershell
$bundlePath = (Resolve-Path -LiteralPath 'G:\reza').Path
$inventory = Get-Content -LiteralPath (Join-Path $bundlePath 'results/provenance/shd-speaker-augmented-export-2026-11-08.json') -Raw | ConvertFrom-Json
foreach ($entry in $inventory.files) {
    $artifactPath = Join-Path $bundlePath $entry.path
    if (-not (Test-Path -LiteralPath $artifactPath -PathType Leaf)) { throw "Missing artifact: $($entry.path)" }
    if ((Get-Item -LiteralPath $artifactPath).Length -ne $entry.bytes) { throw "Size mismatch: $($entry.path)" }
    if ((Get-FileHash -LiteralPath $artifactPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Hash mismatch: $($entry.path)" }
}
Write-Host "Verified $($inventory.files.Count) payload files."
```
