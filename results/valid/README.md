# Canonical reference and model results

Snapshot: 20 October 2026. **21/27 accepted completed runs; 18 matched and 3 accepted historical ECG reference runs.** Six SSC jobs are missing locally: seeds 123 and 456 for reference, DTH and MT.

| Model | SHD* accepted | SSC mean-loss accepted | ECG accepted | Total |
|---|---|---|---|---:|
| Reference | 42, 123, 456 | 42 | 42, 123, 456* | 7/9 |
| DTH | 42, 123, 456 | 42 | 42, 123, 456 | 7/9 |
| MT | 42, 123, 456 | 42 | 42, 123, 456 | 7/9 |

*ECG reference is accepted historical evidence: seed 42 completed 383 epochs, seeds 123/456 stopped early at 283/185 epochs. DTH/MT completed 400. Cross-model ECG comparisons remain descriptive, not fully controlled. Accuracy close to the paper does not erase the deviation.

- `reference/SHD/`: canonical SHD* paper-based reference, with 20% training-derived validation and split seed 42.
- `reference/SSC/`: promoted mean-loss seed 42, 82.22% best validation and 80.69% official test accuracy; diagnostic provenance retained.
- DTH/MT SSC seed 42: complete training and test results, 80.46% and 80.01% test accuracy. Do not confuse these with archived summed-loss scores.
- `reference/ECG/`: the accepted historical reference; per-seed configs remain unchanged.
- DTH/MT ECG: all three full-budget training seeds verified locally.

The official `runner.py` chooses dataset losses automatically: SHD* sums per-timestep softmax, SSC averages membrane potentials, and ECG classifies each labeled timestep. SSC uses `ssc_mean_loss_v1`; SHD/ECG retain their saved baseline protocols. Only the user launches training; agents prepare/check configs and provide commands.

[Per-seed grid](COMPARISON_STATUS.md) and [machine-readable status](COMPARISON_STATUS.json), [SSC comparison](SSC_MEAN_COMPARISON.md), [general report](../../reports/reference-vs-models.md), [paper comparison](../../reports/paper-vs-reference.md).

Historical summed-loss SSC runs remain in `archive/ssc_summed_loss_20261019/`; older early-stopped proposed-model ECG records remain archived separately. Retain manifests and checkpoint identity. Other proposed models remain under [not_improved](../not_improved/README.md), whose category concerns SHD* accuracy only.

## Measurements

Per-seed `COST_RESULTS.json`/`.md` remain beside training summaries. See [measurement layout](MEASUREMENTS.md). DTH/MT new completions have measurements; promoted SSC reference costs remain pending. No measurement was rerun during this reporting update. Use the [official runner guide](../../docs/experiments/ssc-mean-comparison.md) for new SSC training and resume commands.
