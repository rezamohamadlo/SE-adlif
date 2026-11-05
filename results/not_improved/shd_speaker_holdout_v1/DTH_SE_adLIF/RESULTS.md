# DTH_SE_adLIF: SHD speaker holdout (two validation speakers)

Protocol: `shd_speaker_holdout_v1`; module 1.0.0. Baseline: `paper_classification_v1`.
Final official-test accuracy (%) from speaker-validation-selected checkpoints; ordinary inference without TTA.
This protocol differs from random-recording SHD*; paper accuracy is not a matched baseline.
Split fingerprint: `e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
Validation speakers: [0, 3]; validation fraction: 20.046591%.
Per-run resolved settings: `SHD/seed_<seed>/FULL_RUN_CONFIG.json`; split counts/indices: `SPLIT_MANIFEST.json`.

| Dataset / protocol | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD speaker holdout v1 | 92.58 | pending | pending | 1 | 92.58 (SD unavailable) |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD speaker holdout v1 | 1 | 451,484 (n=1) | 450,744 (n=1) | 10,530 (n=1) | 8,866 (n=1) / 1,664 (n=1) | 9.85 (n=1) / 1.85 (n=1) |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD speaker holdout v1 | 1 | 9.97 (n=1) | 0.242895 (n=1) | 0.250372 (n=1) | 0.251353 (n=1) | 2.626 (n=1) |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->

