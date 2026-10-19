# Canonical reference and model results

Snapshot: 19 October 2026.

- `reference/SHD/`: canonical paper-baseline SHD* reference, seeds 42/123/456; batch 256, 300 epochs, early stopping disabled, 20% training-derived validation with split seed 42.
- `reference/SSC/`: promoted mean-loss reference seed 42, 80.69% official test accuracy. Historical summed-loss reference/DTH/MT seeds remain archived; new DTH/MT mean-loss runs are pending.
- `reference/ECG/`: historical unmatched reference: seed 42 used 383 epochs; seeds 123/456 enabled early stopping.
- `DTH_SE_adLIF/` and `MT_SE_adLIF/`: current proposed-model records. Use saved configs and metrics to establish each run's protocol and completion.

`runner.py` applies `config/paper_baseline.yaml` through the dataset-specific comparison group by default and writes to `results/valid/`. `evaluation.yaml` remains a separate historical controlled_v2 record. Older configs may retain former paths and protocol labels; do not rewrite their provenance or infer the protocol from the directory name.

The mean-loss reference diagnostic seed 42 was moved into `reference/SSC/seed_42/`: 82.22% best validation and 80.69% official test accuracy, with original provenance retained. DTH ECG seeds 42/123 and MT ECG seeds 123/456 now have verified full 400-epoch completions; DTH seed 456 and MT seed 42 remain pending. The ECG reference is still unmatched. Historical early-stopped ECG records are archived separately.

Other proposed models and their retained dataset evidence are under [not_improved](../not_improved/README.md). That classification concerns SHD* accuracy only. The completed SHD* comparison is in [the general report](../../reports/reference-vs-models.md); its SSC/ECG section distinguishes incomplete, unmatched and unverified records.

Historical experiment outputs are retained in `archive/experiments_20261017/`; the prior LG-DTH early-stopping run is also retained separately under `archive/`. Preserve archive manifests and checkpoint identity.

## Measurements

Per-seed test activity and compute records live beside training summaries as `COST_RESULTS.json` and `COST_RESULTS.md`. See [the measurement layout](MEASUREMENTS.md). Cost evidence and training completion are separate; this audit did not rerun measurements.

Archiving did not change default loss settings. Use explicitly prepared mean-loss experiment configs for future training rather than assuming empty canonical directories select a new protocol. See the [archive documentation](../../docs/transfers/ssc-summed-loss-archive-2026-10-19.md).

## SSC mean-loss comparison ready

The tested mean-loss reference seed 42 is now in `reference/SSC/seed_42/`: 82.22% validation, 80.69% official test accuracy. Its original diagnostic config/checkpoint bytes and provenance were preserved. `SSC_MEAN_COMPARISON.md` tracks the comparison. Fresh DTH and MT seed-42 configs use `ssc_mean_loss_v1`; launch them yourself with `.\.venv\Scripts\python.exe runner.py --models DTH_SE_adLIF MT_SE_adLIF --datasets SSC --seeds 42 --resume 0`. The official runner shows progress in one terminal and performs validation-selected testing and cost measurements after training. SSC now defaults to mean loss; SHD/ECG defaults are unchanged. See the [guide](../../docs/experiments/ssc-mean-comparison.md).
