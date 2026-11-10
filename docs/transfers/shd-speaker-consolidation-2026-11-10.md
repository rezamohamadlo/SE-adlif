# Physical SHD speaker-results consolidation, 10 November 2026

The user requested removal of the separate experiment folders after combining
results. Full original artifacts now reside in
`results/experiments/shd_speaker_combined_v1/artifacts/`, including all selected
and last checkpoints, original metrics, configs, validation/test records,
source provenance and the separate exploratory baseline TTA evidence.

The former top-level experiment roots are explicitly mapped to identically
named children of `artifacts/` by `config/path_aliases.yaml`:

- `shd_speaker_augmented_v1`
- `shd_speaker_regularization_v1`
- `shd_regularization_test_seed42_v1`
- `shd_speaker_regularization_tests_v1`
- `shd_speaker_regularization_test_v1`
- `shd_speaker_augmented_tta_v1`

The complete suffix below each root is retained. Checkpoint bytes and saved
protocols are unchanged. `project_paths.local_path` resolves old repository-
relative paths and registered old checkout prefixes to the same checkpoint
identity at its consolidated location. Live speaker launchers use canonical
paths while preserving frozen protocol/cache identities and completion guards.

The combined root provides normalized 24-run records and train/validation/test
summaries. `provenance/pre_consolidation/` preserves the former derived indexes
before they were reindexed. Frozen evaluation recipes inside the artifacts
retain their original bytes and source identities; they are historical records.
Current tooling must use the project root and registered path resolution.

The consolidation inventory is
`results/provenance/shd-speaker-consolidation-2026-11-10.json`. Every retained
source file was copied and SHA-256 checked before cleanup. Automatic approval
review rejected the agent's removal command with the reason **blocked by
policy**. The user subsequently removed the four regularization/test folders
locally and deferred further deletion. The separate augmented-baseline and TTA
folders still exist locally; canonical loading uses the consolidated copies.
The subsequent merge of remote `main` preserved its tracked historical files
and restored three of the manually removed roots (`shd_regularization_test_seed42_v1`,
`shd_speaker_regularization_test_v1` and `shd_speaker_regularization_v1`). These
copies remain alongside the canonical collection because further deletion was
cancelled by the user.
Subsequent cleanup: the user later removed all remaining separate roots after
the duplicate audit. All 697 combined files were verified again, including the
distinct remote partial summary preserved under `provenance/remote_main_7ccdc1b/`.
The user requested committing and pushing these manual removals. This final
state supersedes the preceding intermediate folder-status notes.
Only disposable locks/cache/temporary files were
excluded from the preserved artifact copies. External PC1/PC2/PC3 bundles were
not changed or deleted.

The following previously prepared command is optional historical cleanup
guidance. No further deletion is part of the current push request:

```powershell
.\docs\transfers\finish-shd-speaker-cleanup-2026-11-10.ps1
```

The script checks exact workspace targets, verifies every retained file against
the consolidation receipt, rejects unexpected source files and reparse points,
and updates the receipt only after removal succeeds. The agent did not execute
this script or retry the rejected deletion through another tool.

Six original shuffled-run metric CSVs were also verified against matching
checkpoint measurement identities from `G:/reza/SE-adlif-2026-11-10` and copied
to the combined folder's `comparison_evidence/shuffled/`. These supply separate
training and validation values in the speaker report; no metric is inferred
from test accuracy.

For future transfers, copy the combined directory with its `artifacts/` and
provenance children, retain repository-relative layout, and transfer the path
alias configuration through Git. A legacy bundle using the old root names
must be placed into the corresponding `artifacts/<old-name>/` subtree.
Copying it back into the removed top-level directories is unnecessary.

No training, optimizer updates or new inference were performed for this
consolidation. Training, validation and official test accuracies remain
separately labeled; all current comparisons use ordinary inference without TTA.
