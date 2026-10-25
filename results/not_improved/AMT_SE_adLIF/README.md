# AMT: completed three-dataset evidence

Imported on 25 October 2026. Its SHD* three-seed mean did not exceed MT or DTH,
so all dataset folders are retained here under the project's SHD* retention
policy. SSC and ECG outcomes remain separate assessments.

| Dataset | AMT test mean ± sample SD (%) | Difference from MT mean (pp) | Seeds |
|---|---:|---:|---|
| SHD* | 92.71 ± 1.94 | -1.90 | 42, 123, 456 |
| SSC | 80.26 ± 0.18 | +0.18 | 42, 123, 456 |
| ECG | 87.83 ± 0.18 | +0.02 | 42, 123, 456 |

Means and sample SDs use final test accuracy over independent training seeds
(n=3 per dataset); differences use unrounded metrics. These observed differences
do not establish statistical significance. All seeds are retained, including
the lower SHD* seed-123 result. No result was replaced based on test accuracy.

AMT changes MT's fixed mixture into a causal mixture modulated by fast/slow
sample-local activity traces. Trace decays are 0.9/0.99 and new per-neuron gain
parameters start at zero. Other dataset-specific training settings inherit
`paper_classification_v1`; SSC uses the separate `ssc_mean_loss_v1` identity.
See the [experiment guide](../../../docs/experiments/activity-mixed-mt.md).

All nine runs retain full-budget metrics, completion markers, configs,
validation-selected best checkpoints, last checkpoints and complete cost
records. The source [aggregate](RESULTS.md) and original per-run files preserve
their bytes. `config/path_aliases.yaml` resolves the old
`results/experiments/amt_v1/` root to this retention area without substituting
checkpoints or changing their suffixes.

Five local seed-42 startup files with no metrics/checkpoints were copied into
`startup_history/local-seed42-before-import/`; their original filesystem copies
were left in place. That startup attempt is not a completed independent run
and is not the source of the imported seed-42 weights.

[Hash-verified transfer manifest](../../provenance/reza-amt-import-2026-10-25.json).
The external source copies remain at `F:/reza`. Intentional new experiments
need new output directories; do not overwrite or recreate these records.
