# Imported SHD* extension snapshot

Imported on 29 October 2026 from the user-supplied external snapshot `F:/reza/results/experiments/extended_seeds_v1`.

- MT: seven completed 300-epoch training/validation runs, seeds 1009, 2027, 3037, 4049, 5051, 6067, 7079.
- DTH: three completed 300-epoch training/validation runs, seeds 1009, 2027, 3037. The unavailable unfinished run was excluded at the user's instruction.
- All 151 copied files were SHA-256 verified. Effective configs, split seed 42, 20% training-derived validation, selected checkpoints and completed CSV histories were checked.
- Separate existing MT test evaluations, frozen checkpoint plan, evaluation log, script and summary are preserved in `../test_mt_shd_v1/`. The imported script was not executed. Checkpoint hashes and selected identities match the plan. These test results have already been observed.
- Existing ECG snapshots and canonical valid runs were not overwritten. No training or test inference was launched. Source evidence and launch revisions remain unchanged; current-checkout resume compatibility was not established.

[Transfer manifest](../../../provenance/shd-extension-import-2026-10-29.json)

## Completion update ? 31 October 2026

Imported completed DTH seeds 4049, 5051, 6067 and 7079 (300 epochs each).

Both models assigned to this PC now have seven completed extension training/validation runs locally. This does not imply all standalone test evaluations are complete. All 140 source files across the ten newly imported runs were SHA-256 verified, including checkpoints and metrics. No training or inference was launched.

[Completion transfer manifest](../../../provenance/extension-completion-import-2026-10-31.json)
