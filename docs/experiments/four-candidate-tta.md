# Four-candidate frozen-checkpoint TTA

Run the PowerShell commands in `command.txt` from the repository root. No new
packages are required beyond the existing project `.venv` and requirements.
All training remains untouched; the runner dispatches evaluation only.

Models: AMT_SE_adLIF, MT_TC_SE_adLIF, LT_DTH_SE_adLIF, SA_MT_SE_adLIF.
Datasets: SHD (reported as SHD*), SSC, ECG. Training seeds: 42, 123, 456.
Use the unchanged `config/tta.yaml` gains 1.0, 0.95, 1.05 and average class
probabilities. Evaluate validation first and then test without changing this
policy. Existing inspected test scores make this follow-up exploratory.

Checkpoint roots:

- `results/not_improved/AMT_SE_adLIF/`
- `results/not_improved/LT_DTH_SE_adLIF/`
- `results/experiments/sa_mt_v1/SA_MT_SE_adLIF/`
- `results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/`

Transfer the original seed folders including checkpoint weights, configs and
validation metrics, and the required datasets when using another checkout.
Git does not transfer ignored weights or datasets. Use that PC's own `.venv`.
Checkpoint verification checks dataset, cell, seed, full-budget validation
history and validation-selected score. Multiple retained checkpoints require
the completed cost record's exact path and SHA-256 identity. MT-TC uses ordinary
MT inference with its saved weights; its consistency term was training-only.

Outputs are separate from training:
`results/experiments/{amt,mt_tc,lt_dth,sa_mt}_tta_v1/`, with per-seed
`validation/` and `test/` manifests/results and a root `RESULTS.md`.
Rerunning identical commands skips completed evaluations. Manifest mismatches
require a new output directory; do not overwrite existing evidence.

Compare candidate+TTA with MT+TTA and DTH+TTA. This queue measures accuracy,
not latency or energy; three forwards are not a measured threefold runtime.

Preparation on 26 October 2026: all 36 checkpoint/data preflights passed;
no evaluation or training was launched by the agent.
