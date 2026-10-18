# SE-adLIF

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

## Baronig, Ferrand, Sabathiel & Legenstein (2024):
**Advancing Spatio-Temporal Processing in Spiking Neural Networks through Adaptation**

---

## 📌 Getting Started

### Install Dependencies
For a third PC, use the [complete Windows/Codex setup guide](SETUP_THIRD_PC.md)
and `scripts/setup_windows.ps1`. It installs pinned packages with visible
download progress, verifies CUDA/configs, and does not start training.

From the repository root on the new Windows PC:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

For the tested Windows CUDA environment, use Python 3.11:
```powershell
python -m venv .ven
.\.ven\Scripts\python.exe -m pip install -r requirements.txt
```

`environment.yml` retains the original Conda/audio research setup. It is not
the tested Windows environment and includes optional CUDA/JAX dependencies:
```sh
conda env create -f environment.yml
```

Current tables: [paper results](reports/paper-results.md),
[paper versus reference](reports/paper-vs-reference.md), and
[reference versus six proposed models](reports/reference-vs-models.md).
The runner uses `config/paper_baseline.yaml`. Previous guides and tuning notes
are retained in [the report archive](reports/archive/2026-10-15-report-consolidation/).
Current results belong in `results/valid/`; tuning results belong in
`results/experiments/`. Historical notes and document exports are under
`reports/archive/`.

### Automatic activity and cost records

After each newly completed training/test job, `runner.py` evaluates its saved
validation-selected checkpoint and writes `COST_RESULTS.json` and
`COST_RESULTS.md` inside the seed directory. These record total/trainable
parameters, L1/L2 and total spikes, firing fractions, estimated SynOps, dense
matrix and scalar arithmetic GFLOPs, logical operations, and inference timing.
The model's `RESULTS.md` aggregates completed measurements across seeds.
This adds an inference pass and three timing passes; it does not change training.
Definitions and limitations follow the existing measurement reports: logical
operations are not hardware FLOPs/energy, and ECG's analog input is dense work.

Existing completed runs stay skipped by default. Use `--measure-existing-costs`
to backfill selected completed runs (matching saved cost records are reused),
or `--no-measure-costs` to opt out of measurements. `--dry-run` performs no cost
evaluation. For example, using this checkout's environment:

```powershell
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 42 123 456 --dry-run
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 42 123 456 --measure-existing-costs
```

The second command runs any unfinished selected training jobs as well as
collecting costs. To measure just one completed job without scheduling training:

```powershell
.\.venv\Scripts\python.exe measure_run_costs.py --run-dir results/valid/protocol_v2/LG_DTH_SE_adLIF/SHD/seed_456 --dataset SHD
```

A measurement failure writes `COST_ERROR.json`, leaves completed training intact,
and lets the queue continue; its costs stay pending. Unsupported future cells
must add a valid operation-count convention before their costs can be reported.
Runtime means are withheld when hardware/software settings differ across seeds.
Preprocessing/baseline/manifest guards remain unchanged. Manual provenance
sections in model result reports are retained when the runner refreshes tables.

### Working across computers

New classification run manifests and checkpoint configuration fields use
repository-relative artifact paths. `project_paths.py` resolves them against
the checkout root, including when Hydra changes the working directory.
Previous checkout roots are explicitly registered in `config/path_aliases.yaml`;
add another root there if a saved record comes from a different location.
Only the root prefix is relocated: the full checkpoint filename and directory
are retained. Existing historical configs and summaries are not rewritten.

To inspect a portable view of an existing saved config without changing it:

```powershell
.\.ven\Scripts\python.exe project_paths.py --inspect results/archive/experiments_20261017/shd_lr_sweep/c03/SWEEP_CONFIG.json
```

Checkpoints and datasets are not transferred by Git. Copy the exact required
files separately and verify recorded checkpoint hashes. A missing file causes
an error; no newer or similarly named checkpoint is substituted. The standard
classification runner, tuning/search launchers and standalone checkpoint tester
resolve registered historical paths automatically. Already running processes
keep their existing code; these changes apply to subsequent launches.

---

## 🔄 Reproducing Results

### SHD, SCC, ECG, BSD, and Oscillation Experiments
To start an experiment, use:
```sh
python run.py experiment=<experiment_name> ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir
```
**Notes:**
- `datadir` is **mandatory** and should contain the datasets.
- For SHD and SSC, data is downloaded automatically if not found at `datadir/SHDWrapper`.
- BSD and oscillation toy task, datasets are created on the fly, so `datadir` can point to an empty directory.
- Results are stored in a local `results` folder unless `resultdir` is specified.
- `<experiment_name>` refers to configurations in `./config/experiment/`.

### Configuration Overrides (Hydra)
We use [Hydra](https://hydra.cc/) for configuration management. To override parameters, use the `++` syntax. For example, to change the number of training epochs:
```sh
python run.py experiment=SHD_SE_adLIF_small ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir ++n_epochs=10
```
For the BSD task with a different number of classes (Figure 6b):
```sh
python run.py experiment=BSD_SE_adLIF ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir ++dataset.num_classes=10
```

---

## 🎵 Audio Compression Experiments
To start an audio compression experiment, use:
```sh
python run_compress.py experiment=<experiment_name> ++logdir=path/to/my/logdir ++datadir=path/to/my/datadir
```
### Available Configurations
- **SE-adLIF**: `compress_libri_SE_adLIF`
- **EF-adLIF**: `compress_libri_EF_adLIF`
- **LIF**: `compress_libri_LIF`

---

## 📊 Evaluating Audio Compression Models 
Model checkpoints for each configuration are available at [checkpoints](https://github.com/IGITUGraz/SE-adlif/tree/main/checkpoints).
The local CSV measurements from those audio experiments are retained under
`reports/archive/2026-10-15-report-consolidation/audio-compression/`;
they are evaluation records, not model weights.

### 1️⃣ Generating Wave Files
To generate wave files from a checkpoint:
```sh
generate_waves.py ckpt_path=/path/to/ckpt/example.ckpt source_wave_path=/path/to/libritts/location/ pred_wave_path=/path/to/prediction/ encoder_only=$encoder_flag
```
- `$encoder_flag`: `true` or `false`.
- `source_wave_path` can be a single `.wav` file or a directory containing `.wav` files.
- If no valid `.wav` files exist, the clean test-set from LibriTTS (~9h of audio) is used.

### 2️⃣ Evaluating Generated Waves
Use `evaluate_metrics.py` to compute SI-SNR or [Visqol](https://github.com/google/visqol):
```sh
evaluate_metrics.py metric=$metric source_wave_path=path/to/source/waves pred_wave_path=path/to/model/predictions
```
- `$metric` can be `si_snr` or `visqol`.
- **Note:** Visqol must be compiled manually following [these instructions](https://github.com/google/visqol). Additionally, the project requires either `gcc-9`/`g++-9` or `gcc-10`/`g++-10`. Set the compiler using:
```sh
export CC=gcc-9 CXX=g++-9
```
Furthermore, Visqol relies on Bazel but references an outdated HTTP resource (Armadillo) in its WORKSPACE file. 
The ressource has been moved [here](https://sourceforge.net/projects/arma/files/retired/armadillo-9.900.1.tar.xz.RETIRED/download).
You should modify the WORKSPACE file to reference your local copy as instructed [here](https://github.com/google/visqol/issues/117#issuecomment-2407779701).

---

## ⚠️ Important Information

### 🔹 Configuring `main.yaml`
Global parameters (e.g., `device: 'cpu'`, `cuda:0`) can be set in `config/main.yaml`. These settings are used by PyTorch Lightning’s `SingleDeviceStrategy`.

### 🔹 Block Index Padding
For variable-length sequences (e.g., SHD, SSC), a **custom masking procedure** is used:
- **Data vector:** Contains actual data, padded with zeros.
- **Block index (`block_idx`)**: Indicates valid data (1s) and padding (0s).
- **Target vector**: Maps indices to corresponding labels.

#### Example 1: Single Block Targeting
```
data vector: |1011010100101001010000000000000|
             |-----data---------|--padding---|
             ---> time

block_idx:   |1111111111111111111100000000000|
target: [-1, 3]
```
**Explanation:**
- Block `0` has target `-1` (ignored).
- Block `1` has target class `3`.

#### Example 2: Per-Timestep Labeling (ECG Task)
```
data vector: |1 0 1 1 0 0 1 0 0 0 0 0 0|
             |-----data---|--padding---|
             ---> time

block_idx:   |1 2 3 4 5 6 7 0 0 0 0 0 0|
target: [-1, 4, 3, 1, 3, 4, 6, 3]
```
**Explanation:**
- Multiple blocks (`1-7`) have corresponding target labels.
- Padding (`0s`) is ignored during loss computation.

Using this method, **per-block predictions** can be efficiently gathered using `torch.scatter_reduce`, ignoring padded time steps.

---



This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
