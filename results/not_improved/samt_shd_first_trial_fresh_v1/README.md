# SA-MT SHD* fresh exploratory trial

Imported on 2026-11-02 from external storage `F:/reza/samt_shd_first_trial_fresh_v1`.
All 40 research files were verified byte-for-byte with SHA-256, including six
best/last checkpoints. The source was preserved. The machine-local `.search.lock`
was excluded; it is not transferable evidence of a held launcher lock.

## Tested configuration

Fresh random weights and optimizer, seeds 42/123/456; SA-MT with two 360-neuron
layers; LR 0.0004; dropout 0.10; batch 256; 300 epochs; no early stopping;
plateau scheduler factor 0.9 and patience 10. SHD* reserves 20% of the official
training set for validation with split seed 42; loss is `sum_softmax_over_time`.
Saved manifests identify `paper_classification_v1` but deliberately override
optimization settings. This is not a matched default-baseline comparison.

| Training seed | Best validation accuracy (%) | Recorded test accuracy (%) |
|---|---:|---:|
| 42 | 96.44 | 90.28 |
| 123 | 96.38 | 86.62 |
| 456 | 95.34 | 89.31 |

Test mean +/- sample SD: **88.74 +/- 1.90%**, n=3 independent training seeds.
Values were checked against each `metrics.csv`; all have 300 validation entries
and one test entry. The test summary agrees with those unrounded metrics.
Best and last checkpoint files are retained; checkpoint contents and the original
test-loading procedure have not been independently audited or re-evaluated.

## Permitted interpretation

The source manifest states that this configuration was chosen after inspecting
continuation test scores. Retain it as exploratory evidence that this particular
configuration's favorable continuation behavior did not translate to strong
fresh-training test accuracy. It helps avoid duplicate experiments and supports
analysis of validation/test differences, without establishing their cause.

These results do not demonstrate improvement over DTH/MT and are retained here
under the repository's non-improvement policy. Different optimization settings
prevent a fair architecture-only conclusion. Do not mix them into the main
matched baseline grid, the SSC study, or independent confirmation results.
Do not select further candidates using these test scores.

Original manifests, summaries, generated configs, metrics and checkpoint paths
remain unchanged as provenance. The original experiment-relative paths embedded
in them refer to the former location; do not substitute another checkpoint if
a consumer cannot resolve them. A future standalone evaluation must explicitly
resolve the retained checkpoint identity and its saved config first.
