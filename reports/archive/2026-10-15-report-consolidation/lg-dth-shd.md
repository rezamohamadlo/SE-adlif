# LG-DTH-SE-adLIF on SHD*

The experimental `LGDTHSEAdLIF` cell inherits DTH-SE-adLIF's state initialization,
scan interface, constraints, membrane/adaptation dynamics and surrogate gradient.
Fast and slow traces now track each neuron's previous spikes. Their neuron means
provide global activity; centered local slow and transient terms correct the
original global threshold logit. No batch reduction or labels enter this update.
Both raw per-neuron coefficients start at zero; effective coefficients are
`5 * tanh(raw)`. The positive finite bound is configurable per hidden layer with
`local_coefficient_bound`. The two enable flags control the corrections separately.

All three configs inherit SHD_DTH_SE_adLIF, retaining two recurrent 360-neuron
layers and the reference preprocessing, dropout and loss. `runner.py` applies
the unchanged `controlled_v2` protocol: training-derived 20% validation, split
seed 42, 300 epochs, batch 512, Adam LR 0.01, clipping 1.5, validation accuracy
checkpoint selection and early stopping. Official test evaluation follows
selection. No full SHD training or official test evaluation was run here.

Trainable parameters (entire MLPSNN): DTH 450,744; LG-DTH 452,184.
The increase is 1,440 (two coefficients for each of 720 hidden neurons).
Ablations retain the same checkpoint layout and parameter count; disabled
coefficients do not enter the forward graph and receive no gradient.

Validation: all 52 tests in `tests/` passed, including six LG-DTH tests. The
short synthetic smoke ran on the NVIDIA RTX 4090 with the full hidden architecture,
one training and validation batch, checkpoint saving/loading, then one resumed
training and validation batch. The resumed global step was 2 and Adam state was
restored. Runner dry-run validated all nine variant/seed plans without writes.

Exact initialization tests compare spikes, membrane and adaptation states with
zero tolerance against the DTH model. With learned nonzero global gains, averaging
local traces and recursively updating an already averaged trace are mathematically
equivalent but may differ by floating-point rounding.

Run these PowerShell commands from the repository root. They are prepared commands,
not jobs launched during implementation. The both-correction ablation is the main
model. Each model variant has its own directory. Rerunning resumes `last.ckpt` or
skips completed runs; changed configurations are rejected by the existing runner.

```powershell
# Seed 42: both local corrections (main model / both-correction ablation)
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 42

# Seed 42: local slow only
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF_slow --datasets SHD --seeds 42

# Seed 42: local transient only
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF_transient --datasets SHD --seeds 42

# Additional main-model seeds, not executed
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 123
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF --datasets SHD --seeds 456

# Validate all planned variants/seeds without training or writes
.\.venv\Scripts\python.exe runner.py --models LG_DTH_SE_adLIF LG_DTH_SE_adLIF_slow LG_DTH_SE_adLIF_transient --datasets SHD --seeds 42 123 456 --dry-run

# Unit tests including a short synthetic GPU training/save/resume smoke test
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Future run directories:

```text
results/valid/protocol_v2/<model>/SHD/seed_<seed>/
  ckpt/                         validation-selected checkpoints and last.ckpt
  logs/mlp_snn/version_0/        metrics.csv
  FULL_RUN_CONFIG.json           resolved configuration / resume guard
  completed.txt                 written after training and final test evaluation
results/valid/protocol_v2/<model>/RESULTS.md
```

Here `<model>` is `LG_DTH_SE_adLIF`, `LG_DTH_SE_adLIF_slow`, or
`LG_DTH_SE_adLIF_transient`, and `<seed>` is 42, 123, or 456. These training output
directories are not created by the dry run. Smoke checkpoints use temporary
directories and are cleaned up. Existing result artifacts are preserved.
