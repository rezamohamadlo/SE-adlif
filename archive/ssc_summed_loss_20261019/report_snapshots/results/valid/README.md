# Canonical reference and model results

Snapshot: 19 October 2026.

- `reference/SHD/`: canonical paper-baseline SHD* reference, seeds 42/123/456; batch 256, 300 epochs, early stopping disabled, 20% training-derived validation with split seed 42.
- `reference/SSC/`: official train/valid/test reference; three completed 40-epoch runs.
- `reference/ECG/`: historical unmatched reference: seed 42 used 383 epochs; seeds 123/456 enabled early stopping.
- `DTH_SE_adLIF/` and `MT_SE_adLIF/`: current proposed-model records. Use saved configs and metrics to establish each run's protocol and completion.

`runner.py` applies `config/paper_baseline.yaml` through the dataset-specific comparison group by default and writes to `results/valid/`. `evaluation.yaml` remains a separate historical controlled_v2 record. Older configs may retain former paths and protocol labels; do not rewrite their provenance or infer the protocol from the directory name.

DTH and MT each have three completed summed-loss SSC runs with metrics and checkpoints. DTH: 78.19 +/- 0.14%; MT: 77.73 +/- 0.36%. The original seeds 123/456 were moved from external storage on 19 October 2026; prior local snapshots are archived. DTH ECG seeds 42/123 and MT ECG seeds 123/456 now have verified full 400-epoch completions; DTH seed 456 and MT seed 42 remain pending. The ECG reference is still unmatched. Historical early-stopped ECG records are archived separately.

Other proposed models and their retained dataset evidence are under [not_improved](../not_improved/README.md). That classification concerns SHD* accuracy only. The completed SHD* comparison is in [the general report](../../reports/reference-vs-models.md); its SSC/ECG section distinguishes incomplete, unmatched and unverified records.

Historical experiment outputs are retained in `archive/experiments_20261017/`; the prior LG-DTH early-stopping run is also retained separately under `archive/`. Preserve archive manifests and checkpoint identity.

## Measurements

Per-seed test activity and compute records live beside training summaries as `COST_RESULTS.json` and `COST_RESULTS.md`. See [the measurement layout](MEASUREMENTS.md). Cost evidence and training completion are separate; this audit did not rerun measurements.
