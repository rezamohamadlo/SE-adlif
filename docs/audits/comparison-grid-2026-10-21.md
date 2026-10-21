# Current comparison and reporting audit - 21 October 2026

The reference/DTH/MT grid now contains 27 accepted completed runs: seeds
42/123/456 for SHD*, SSC and ECG. Of these, 24 have matched settings; the
three ECG reference runs remain accepted historical evidence with different
stopping/budgets. No training or inference was performed for this audit.

SSC test means and sample SDs across three training seeds are reference
80.47 +/- 0.20%, DTH 79.95 +/- 0.46% and MT 80.07 +/- 0.14%.
Means and differences use unrounded saved metrics. Reference seed 42 retains
its original diagnostic provenance. Six imported runs have hash-verified
selected checkpoints and last checkpoints at epoch 39 with optimizer state.
All 27 seed directories have saved cost records.

The project README, results README, model summaries, comparison status,
SSC comparison, reference aggregates, main comparison reports and reference
sparsity figure now reflect the three-seed SSC results. Active reference cost
tables use mean-loss checkpoints; summed-loss evidence remains archived.
The SSC and TTA guides now distinguish completed training/checkpoint
availability from pending standalone evaluations. Checkpoint availability
does not establish completion of a TTA evaluation.

Existing TTA summaries were regenerated from their saved result records,
checking checkpoint identities against manifests and local checkpoint hashes.
Validation and test records remain separate; no missing TTA results were
inferred. Historical audits, transfer manifests, archived tables and original
per-run evidence retain their original values and snapshot context.

Verification: eight TTA unit tests and 42 runner, cost and project-path tests
passed. Current report/guide links resolve locally; Git whitespace checks pass.
The configured `.ven` interpreter is absent in this checkout, so verification
used the existing project `.venv` interpreter without installing packages.
Ignored checkpoints, metrics and datasets remain local and are not transferred
by Git push.

[Current grid](../../results/valid/COMPARISON_STATUS.md) |
[SSC comparison](../../results/valid/SSC_MEAN_COMPARISON.md) |
[Import provenance](../../results/provenance/ssc-local-import-2026-10-21.json)
