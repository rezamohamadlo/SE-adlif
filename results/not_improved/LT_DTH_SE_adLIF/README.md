# LT-DTH: completed three-dataset evidence

Imported on 25 October 2026. The SHD* three-seed mean did not exceed DTH or MT,
so the complete model record is retained here under the project policy.
This classification concerns SHD*; SSC and ECG retain their separate outcomes.

| Dataset | LT-DTH test mean ± sample SD (%) | Difference from MT mean (pp) | Completed seeds |
|---|---:|---:|---|
| SHD* | 93.98 ± 0.63 | -0.63 | 42, 123, 456 |
| SSC | 80.17 ± 0.19 | +0.10 | 42, 123, 456 |
| ECG | 87.67 ± 0.61 | -0.14 | 42, 123, 456 |

Values use final test metrics with matching training seeds (n=3); SD is sample
SD and differences use unrounded metrics. These observed differences do not
establish statistical significance. The primary DTH SHD* mean is 94.49%, versus
93.98% here. No records were selected or replaced based on their test scores.

The model learns two layer-level activity-trace decay rates per hidden layer,
initialized at DTH's 0.9/0.99. The fast range is [0.5, 0.98] and the slow range
is [0.985, 0.9995]. Saved configs retain the dataset-specific paper baseline:
SHD* 300 epochs/batch 256, SSC 40/256, ECG 400/64, without early stopping.
SSC uses mean membrane-potential loss. See the
[experiment guide](../../../docs/experiments/learned-trace-dth.md).

All nine runs include metrics, completion markers, saved configs, validation-
selected best checkpoints, resume checkpoints and complete per-seed cost records.
The imported [aggregate](RESULTS.md) and underlying records retain their original
bytes. Original paths under `results/experiments/lt_dth_v1/` resolve to this
location through `config/path_aliases.yaml`; no checkpoint was substituted.
The original external copies remain on `F:/reza`.

[Hash-verified transfer manifest](../../provenance/reza-results-import-2026-10-25.json).
Intentional reruns need a new configuration/output directory; do not recreate
or overwrite these completed runs.
