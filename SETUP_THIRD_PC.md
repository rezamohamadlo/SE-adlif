# Third-PC setup: Windows, NVIDIA GPU, Python 3.11

This guide prepares another PC for future classification jobs with the same
repository configuration. It does not schedule training, merge results, install
drivers, or change the paper baseline. Use the same Git commit on all PCs and
the exact package pins in `requirements.txt`.

The user selected the **repository pins**, including **NumPy 1.26.4**. During
preparation on 18 October 2026, the source PC was found to have NumPy 2.4.6 despite
that pin; Tonic 1.6.0 declares NumPy <2. This guide deliberately creates the clean
pinned environment, not a copy of that drift. Existing result provenance remains
unchanged. The source interpreter was Python 3.11.0 x64; the bootstrap accepts
3.11.x x64 and records the exact patch version. Same configuration and seeds do
not guarantee bit-identical scores on different GPUs, drivers or Python patches.

## 1. Prerequisites on the new PC

Use Windows x64 with a compatible NVIDIA GPU. Install Git for Windows and 64-bit
Python 3.11 with its Windows launcher. If missing, Codex can use the following
commands in an interactive terminal (system installation may require approval):

```powershell
winget install --id Git.Git --exact --source winget
winget install --id Python.Python.3.11 --exact --source winget
```

Open a new terminal after installation and check:

```powershell
git --version
py -3.11 -c "import sys,struct; print(sys.version); print(struct.calcsize('P')*8)"
nvidia-smi
```

The Python command must show 3.11 and 64 bits. Install/update the NVIDIA driver
from its official source if needed, then reboot when requested. A CUDA number
printed by `nvidia-smi` is not the PyTorch wheel version: retain **PyTorch
2.6.0+cu124**, torchvision **0.21.0+cu124**, and torchaudio **2.6.0+cu124**.
The bootstrap tests an actual CUDA matrix operation, not just GPU detection.
If this GPU cannot execute the pinned wheel, stop and report it; do not silently
upgrade PyTorch, switch to CPU, reduce batch size, or alter training settings.

Official references: [Git for Windows](https://git-scm.com/downloads/win),
[Python Windows launcher](https://docs.python.org/3.11/using/windows.html#python-launcher-for-windows),
[PyTorch previous-version installation](https://pytorch.org/get-started/previous-versions/),
[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

## 2. Get identical source code

First commit/push the intended source-PC changes, including this guide and its
scripts. Record the full commit on the source PC:

```powershell
git status --short
git rev-parse HEAD
```

On the third PC, choose any writable directory; the original drive/username is
not required. Example:

```powershell
git clone https://github.com/rezamohamadlo/SE-adlif.git
Set-Location SE-adlif
git pull --ff-only
git rev-parse HEAD
```

Compare the full commits. If a specific committed revision is required, use
`git switch --detach FULL_COMMIT_HASH` before setup. Do not reset, clean, rebase
or overwrite an existing checkout with local edits. Authentication is handled
through Git's normal login flow, not credentials pasted into scripts.

## 3. Bootstrap with live installation progress

From the checkout, preview the plan:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -PlanOnly
```

Then run it, replacing the example hash with the full source-PC commit:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
    -ExpectedCommit FULL_COMMIT_HASH
```

`ExecutionPolicy Bypass` applies only to this PowerShell process. No persistent
execution-policy change is made. The script resolves paths from its own location,
so it also works when called by absolute path from another working directory.

Run this in a visible PowerShell/VS Code terminal to watch downloads live.
The script creates **`.venv`**, checks Python 3.11 x64, installs the source PC's
pip toolchain version **26.2.1**, then installs `requirements.txt`. Every package
is pinned; the CUDA wheel index is already in that file. Installer output remains
visible, with verbose download/install status and progress bars. Detailed pip
logs are saved under `results/setup/pip-<timestamp>.log`.

The equivalent visible package command after creating the environment is:

```powershell
.\.venv\Scripts\python.exe -m pip --log setup-pip.log install `
    --verbose --progress-bar on -r requirements.txt
```

To watch a recorded log from a second terminal, use the actual filename:

```powershell
Get-Content -LiteralPath results/setup/pip-ACTUAL_TIMESTAMP.log -Wait -Tail 30
```

There is no stdout/stderr redirection or `--quiet` flag hiding download status.
If the network is too slow or a pinned wheel is unavailable, report the exact
failed package, interpreter, URL/error and log location. Do not loosen pins or
install into global Python to make the command pass.

For an explicitly chosen Python executable, add `-PythonExe 'C:\path\python.exe'`.
The default environment directory is `.venv`. Do not copy another machine's
virtual environment or delete/recreate an existing environment automatically.

## 4. Verify packages, CUDA and effective configs

Bootstrap automatically runs:

- `pip check` and an exact installed-version comparison against requirements.
- Key compatibility imports, a real CUDA matrix kernel, GPU/device recording.
- Resolved reference configs for SHD, SSC and ECG and baseline LR/batch/budget checks.
- Runner dry-runs for the reference on all three datasets and MT-DTH on SHD*.

The report is `results/setup/SETUP_REPORT.json`: package versions, errors,
Git revision, all config-file hashes, resolved dataset configs, GPU details,
and a separate data/checkpoint inventory. **Environment readiness is not data
readiness.** Missing datasets do not trigger downloads during setup.
Config hashes normalize Git's CRLF/LF conversion so line endings alone do not
look like changed training settings. Tracked source/config edits are reported
and must be resolved without discarding work before matching the source commit.

To check an already installed environment without reinstalling packages:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 `
    -VerifyOnly -ExpectedCommit FULL_COMMIT_HASH
```

Optional full repository unit tests:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -VerifyOnly -RunTests
```

Tests include tiny synthetic fit/resume smoke tests in temporary directories;
they never start the actual 300/400-epoch experiment suite. They can use the GPU,
so run them before starting expensive work. Failure must be investigated rather
than hidden by changing baseline/model code. Fresh installation on another PC
is not considered verified until that PC's checks pass.

## 5. Copy ignored data and artifacts separately

Git transfers code, tracked configs and reports. It does **not** transfer the
virtual environment, datasets, caches, checkpoints, `completed.txt`, or all logs.
Copy the source PC's `data/` directory into the same repository-relative location:

| Dataset | Required raw extracted files |
|---|---|
| SHD* (config ID `SHD`) | `data/SHDWrapper/shd_train.h5`, `shd_test.h5` |
| SSC | `data/SSCWrapper/ssc_train.h5`, `ssc_valid.h5`, `ssc_test.h5` |
| ECG | `data/QTDB/QTDB_train.mat`, `QTDB_test.mat` |

Copying the whole `data/` also preserves archives and preprocessing caches.
Existing caches are specific to transforms/splits; never relabel stale caches
for different settings. The baseline's fixed split seed is 42. If reconstructing
caches from raw data, preserve preprocessing and verify evaluation consistency.

For existing models, copy the full **completed** `results/` trees (including
ignored checkpoint/log/marker files) while preserving relative layout. For an
interrupted job, stop it on its source PC first and copy its complete snapshot;
resume from its own `last.ckpt`. Final evaluation uses its validation-selected
best checkpoint. A Git clone's tracked summaries alone are not a resumable run.
Never run the same output directory concurrently on two PCs or copy live files
into a running job. Do not mirror/delete local results to make folders match.
Compare differing same-path files and ask before resolving conflicts.

Compare SHA-256 hashes for copied datasets and checkpoints when checking transfer:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath data/SHDWrapper/shd_test.h5
# Run the same command on the source PC and compare the digest.
```

Checkpoint hashes in `COST_RESULTS.json` and reference records identify the exact
files. A missing checkpoint must be copied, never replaced by another filename.
`project_paths.py` resolves repository-relative paths from the checkout root.
Only explicitly registered old prefixes in `config/path_aliases.yaml` are
relocated; ask before adding an unknown old prefix, and preserve checkpoint
identity. New runs should save relative paths without usernames/drive letters.

## 6. Preserve the same comparison settings

Authority: `config/paper_baseline.yaml`, currently `paper_classification_v1`.
Use its dataset section and the runner's `config/comparison/` inheritance.

| Dataset | Batch | Epoch budget | LR | Validation |
|---|---:|---:|---:|---|
| SHD* | 256 | 300 | 0.01 | 20% of official training data, split seed 42 |
| SSC | 256 | 40 | 0.006 | Official validation split |
| ECG | 64 | 400 | 0.01 | 5% of training data, split seed 42 |

Training seeds: 42, 123, 456. Early stopping is disabled in the shared baseline.
Do not change LR, batch, epochs, preprocessing, split, dropout, architecture,
stopping rules or output manifests because the third PC has different hardware.
If the GPU cannot accommodate the baseline batch, report it before deciding on a
separate experiment. `evaluation.yaml` is the historical controlled protocol;
do not pass it over the new baseline or mix its outputs into matched comparisons.

For future jobs, first use a dry-run with the **actual intended model and paths**.
A fresh clone already includes some tracked run reports; without ignored
checkpoints/markers the default directories may correctly fail resume guards.
Transfer the complete existing run, or choose an explicitly new output root for
a fresh experiment. Never delete canonical results to bypass those guards.

Example preflight for a deliberately separate fresh baseline run:

```powershell
.\.venv\Scripts\python.exe runner.py --models MT_SE_adLIF --datasets SHD `
    --seeds 42 123 456 --logdir results/valid/new_baseline_runs --dry-run
```

Only after the user explicitly requests that job, remove `--dry-run` to launch it.
Repeating an existing seed is a rerun, not a new independent training seed.
MT-DTH currently has a SHD-only config: always specify `--datasets SHD` for it.
The runner automatically records post-test spikes, parameters, operation costs
and timing for newly completed runs; hardware/software details remain part of
that evidence. Do not combine runtimes from different PCs as a matched benchmark.

## 7. Prompt to paste into Codex on the third PC

Replace `FULL_COMMIT_HASH` before sending this prompt:

```text
Set up this SE-adlif checkout on this Windows/NVIDIA PC for future training.
Read AGENTS.md and SETUP_THIRD_PC.md and follow scripts/setup_windows.ps1.
The source-PC commit must be FULL_COMMIT_HASH. Preserve local edits and verify
the checked-out commit before installing. Install missing Git/Python prerequisites
through official sources as needed, create the repository-local .venv with Python
3.11 x64, and use repository requirements.txt pins (including NumPy 1.26.4 and
PyTorch 2.6.0+cu124). Keep pip verbose output and progress bars visible, save its log,
and show me that log's path. Use a visible interactive PowerShell/VS Code terminal;
do not hide the installer or redirect away its output. Do not install into global
Python or copy an old venv.
Run package, CUDA-kernel and config checks plus runner dry-runs; do not start any
real training job, test-set benchmark, automatic data download, or results merge.
Inventory missing raw datasets/checkpoints separately from environment readiness.
Never silently change pins, use CPU fallback, alter baseline settings, weaken
manifest guards, substitute checkpoints, or overwrite conflicting copied artifacts.
If a prerequisite needs an interactive installer/reboot, or a pinned dependency
cannot be installed, report the exact blocker and continue independent setup work.
Return the commit, interpreter, pinned package check, GPU verification, baseline
identity/config hashes, data readiness and results/setup/SETUP_REPORT.json path.
Provide a training command only after the intended model/dataset/output is known.
```

The bootstrap was prepared and locally checked without reinstalling the source
PC environment. Driver installation and a clean download/install must still be
verified on the destination PC; the NumPy drift on the source PC is not an
excuse to claim that its live environment passes the new pinned-version check.
