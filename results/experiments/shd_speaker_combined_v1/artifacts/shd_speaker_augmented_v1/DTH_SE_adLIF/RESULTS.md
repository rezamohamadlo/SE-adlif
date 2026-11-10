# DTH_SE_adLIF: SHD* speaker holdout + augmentation (two validation speakers)

Protocol: `shd_speaker_augmented_v1`; module 1.0.0. Baseline: `paper_classification_v1`.
Final official-test accuracy (%) from speaker-validation-selected checkpoints; ordinary inference without TTA.
This protocol differs from random-recording SHD*; paper accuracy is not a matched baseline.
Split fingerprint: `e5a914f492e466b338b76e254dac5b404fe3cffd459b14a2973425266a08674a`.
Validation speakers: [0, 3]; validation fraction: 20.046591%.
Per-run resolved settings: `SHD/seed_<seed>/FULL_RUN_CONFIG.json`; split counts/indices: `SPLIT_MANIFEST.json`.

| Dataset / protocol | Seed 42 | Seed 123 | Seed 456 | n | Mean ± sample SD |
|---|---:|---:|---:|---:|---:|
| SHD* speaker holdout + augmentation v1 | 92.84 | 93.07 | 93.51 | 3 | 93.14 ± 0.34 |

<!-- runner-costs:start -->
## Automatically saved test-set activity and cost

Mean ± sample SD across measured completed training seeds. n=1 has no sample SD. Missing measurements remain pending.

| Dataset | n | Total parameters | Trainable parameters | Hidden spikes/sample | Spikes/sample L1 / L2 | Firing fraction (%) L1 / L2 |
|---|---:|---:|---:|---:|---|---|
| SHD* speaker holdout + augmentation v1 | 3 | 451,484 ± 0 | 450,744 ± 0 | 12,682 ± 150 | 10,765 ± 151 / 1,918 ± 109 | 11.95 ± 0.17 / 2.13 ± 0.12 |

| Dataset | n | SynOps/sample (M) | Dense matrix GFLOPs/sample | Matrix + scalar arithmetic GFLOPs/sample | Logical GOp/sample | Forward ms/sample |
|---|---:|---:|---:|---:|---:|---:|
| SHD* speaker holdout + augmentation v1 | 3 | 11.43 ± 0.10 | 0.242895 ± 0.000000 | 0.250372 ± 0.000000 | 0.251353 ± 0.000000 | 1.791 ± 0.027 |

Per-seed evidence: `<dataset>/seed_<seed>/COST_RESULTS.json` and `COST_RESULTS.md`. Spikes use stored sequence lengths including minimum padding/burn-in; compute/runtime includes batch padding. ECG SynOps exclude analog input, whose dense MACs are stored separately. Arithmetic excludes nonlinearities/comparisons; logical GOp is not hardware GFLOPs or energy. Runtime covers uninstrumented forward only, excluding transfer/loading/loss/temporal aggregation. Future measurement does not change training settings or validate baseline compatibility.
<!-- runner-costs:end -->
