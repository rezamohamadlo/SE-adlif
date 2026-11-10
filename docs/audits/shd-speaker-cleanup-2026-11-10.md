# SHD speaker collection: duplicate-folder audit

Scope: the speaker-holdout augmentation and regularization collection after
remote integration. This is an inspection, not a deletion or a new experiment.

Keep `results/experiments/shd_speaker_combined_v1/` as the canonical collection.
All 420 original retained artifacts pass their recorded SHA-256 checks,
including checkpoint and metric files. No active Python process was found
during inspection.

The following separate folders under `results/experiments/` are redundant:

| Folder | Files present | Approximate size (MiB) |
|---|---:|---:|
| `shd_speaker_augmented_v1` | 96 | 64.18 |
| `shd_speaker_augmented_tta_v1` | 15 | 0.06 |
| `shd_regularization_test_seed42_v1` | 12 | 0.005 |
| `shd_speaker_regularization_test_v1` | 9 | 0.009 |
| `shd_speaker_regularization_v1` | 89 | 1.88 |

Of these 221 files, 212 match their canonical artifact copies byte for byte,
eight are disposable lock files, and one is a historical partial validation
summary restored by the remote merge. That distinct summary is now preserved
byte for byte under the collection's
`provenance/remote_main_7ccdc1b/shd_speaker_regularization_v1/VALIDATION_COMPARISON.md`,
with `SOURCE_VARIANTS.json` recording its original location and hash. The complete
canonical validation summary and all numeric evidence remain unchanged. The
collection inventory includes both added provenance files.

`shd_speaker_regularization_tests_v1` is already absent at the top level; its
original PC2 evidence remains inside the canonical collection. The obsolete
`reports/comparison.md` and `reports/shd-speaker-regularization/comparison.md`
are also absent, despite their IDE tabs still being open.

Live loading uses the registered path aliases. The stale source link in
`reports/comparison-summary.md` now targets the collection's README. Keep
`config/path_aliases.yaml`, experiment configs, source/evaluation tools and tests;
these reproduce or resolve the evidence and are not redundant result copies.
Keep the collection's `artifacts/`, `evidence/`, `comparison_evidence/` and
`provenance/` children: their roles differ, and historical snapshots must remain.
Other experiments, canonical reference results, datasets and caches are outside
this duplicate-folder audit.

After this inspection, the user manually removed all remaining separate roots.
The combined inventory was rechecked: all 697 files passed size and SHA-256
verification. The 159 tracked-file deletions are included in the subsequent
user-requested cleanup commit so that a normal pull receives the consolidated
layout. No agent deletion or execution of the manual cleanup helper occurred;
the earlier automatic approval rejection remains a historical record.
