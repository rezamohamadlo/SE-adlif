# Restore missing local research artifacts from the full results backup

Source: external storage `G:/reza/results`. Destination: repository-relative
`results/`. The source was read only; existing local files were preserved.
No training, checkpoint evaluation, deletion, Git commit or push was performed.

Subsequently, the user requested Git delivery of the 42 local-only archival
research files. These specific files are explicitly tracked despite the general
ignore rules for generated logs/configs. Narrow `.gitattributes` rules disable
line-ending conversion for this historical evidence, preserving the recorded
SHA-256 identities on other checkouts. Their total size is 380,016 bytes.
This Git update also includes the transfer documentation, helper and provenance;
checkpoints remain excluded and still require separate artifact transfer.

Before copying, local results contained 1,773 files and the external tree
contained 3,049. Registered aliases were applied using `project_paths.py`.
The backup had 342 checkpoints; 54 were already local and matched byte for byte.
Shared files were either identical (1,633) or differed only in line endings (97).
No model/configuration conflict blocked the import.

The comparison found **42 local research files absent at their corresponding
external paths**, all in the six `augmented_v1` runs beneath the combined speaker
collection, plus one disposable Python bytecode file. Each run has seven such
files: `.hydra/config.yaml`, `.hydra/hydra.yaml`, `.hydra/overrides.yaml`,
`completed.txt`, `logs/mlp_snn/version_0/hparams.yaml`,
`logs/mlp_snn/version_0/metrics.csv` and `out.log`. Models are DTH/LG-DTH and seeds
are 42, 123 and 456. Twelve of these research files have identical content
elsewhere in the external tree; 30 do not. All local-only files remain intact.
The per-file paths, hashes and alternate content matches are in the manifest.

Imported **1,244 files, including 288 checkpoints and 956 supporting files**,
totaling 2,530,415,982 bytes (about 2.36 GiB). Every copy passed a source/destination
SHA-256 check. All 112 missing checkpoints with hashes recorded in existing
local run evidence matched those expected identities. Other imported files,
including remaining checkpoints, were verified against their external originals.
There are now **342 local checkpoint files**.

Two legacy AMT files collided with newer canonical files after alias resolution:
`.hydra/hydra.yaml` and `out.log` for SHD seed 42. The canonical source files were
imported to the current run directory; both distinct legacy versions were
preserved under
`results/provenance/reza-full-results-sync-2026-11-10/source_variants/`.
All 72 external lock files were skipped. No obsolete separate speaker-result
root was recreated. Two additional historical provenance files from the
external combined collection were also restored; their hashes are recorded in
this transfer manifest, while the existing collection manifest is unchanged.

Machine-readable inventory and receipt:
[`results/provenance/reza-full-results-sync-2026-11-10.json`](../../results/provenance/reza-full-results-sync-2026-11-10.json).
It records both inventories, local-only files, alias collisions, checkpoint
identity checks, every copied path/hash and final verification. The one-time
helper is [`sync-reza-results-2026-11-10.py`](sync-reza-results-2026-11-10.py);
its completed receipt intentionally prevents accidental replay. Checks were
file/inventory/hash validation and Python syntax validation, with no training.
