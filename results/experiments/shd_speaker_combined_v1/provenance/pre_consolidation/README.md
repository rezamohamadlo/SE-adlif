# SHD speaker-based experiments: combined result collection

This folder brings together the augmented dropout-0.25 baseline and all three
regularization variants for DTH and LG-DTH, with seeds 42, 123 and 456.
It contains **24 completed runs** and **8 matched three-seed summaries**.

- [Results and per-seed table](RESULTS.md)
- [Machine-readable results](RESULTS.json)
- [Flat CSV of all 24 runs](RUNS.csv)
- [Validation-based settings selection](VALIDATION_SELECTION.json)
- [Source paths, hashes and snapshot inventory](SOURCE_MANIFEST.json)
- [Frozen speaker split](SPLIT_MANIFEST.json)
- [Paper/reference/shuffled comparison](../../../reports/speaker-based-summary.md)

Each `<profile>/<model>/SHD/seed_<seed>/` contains a normalized `RESULT.json`
and an `evidence/` directory with byte-identical copies of the original config,
metrics, summaries and applicable validation/test/measurement records. The four
profiles are `augmented_v1`, `dropout030_v1`, `dropout035_v1` and
`spike_delete010_v1`. All have the same held-out validation speakers 0 and 3,
split seed 42, training seeds 42/123/456, batch 256 and 300-epoch budget.

Checkpoint files remain at their recorded original locations. Each result
contains repository-relative paths and SHA-256 identities for both the selected
checkpoint and its own last checkpoint. Resolve these paths with
`project_paths.py`; keep the original relative layout when transferring weights.
The snapshots are an audited result view, not a new resumable training run.
Original saved configs, protocols, directories and evaluation scripts remain
authoritative. No original evidence is moved or relabeled.

JSON/CSV accuracies are fractions; Markdown displays percentages. SD is sample
SD across the matching three seeds. Train accuracy at the selected epoch comes
from `metrics.csv`, because historical training summaries can lag. All test
values are ordinary inference without TTA. The regularization test results
were observed before the merged settings selection and remain exploratory.
Settings selection uses validation means only, including the original baseline.

`BUILD_COLLECTION.py --dry-run` checks local configs, full validation histories,
checkpoint identities and test provenance without training or inference. Running
it without that flag writes the collection and rejects changed existing output.
New or changed source evidence requires a new versioned collection directory.
