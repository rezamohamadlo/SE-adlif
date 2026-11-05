# SHD speaker holdout changelog

## 1.0.1 — 5 November 2026

- Retire this workflow at the user's request and reject accidental relaunch.
- Archive the complete result tree, including the interrupted DTH seed-123 run,
  under `results/not_improved/shd_speaker_holdout_v1/` with a SHA-256 inventory.
- Add a directory alias for historical artifact paths; preserve original saved
  configs, metrics, manifests and checkpoint bytes.
- Keep the incomplete comparison explicitly exploratory; retirement is a user
  decision and does not establish an architectural or statistical conclusion.

## 1.0.0 — 4 November 2026

- Introduce protocol `shd_speaker_holdout_v1` as an opt-in shared-runner workflow,
  separately versioned from the paper baseline and random-recording SHD*.
- Freeze two validation speakers using seed 42 and stable SHA-256 ID ranking;
  verify real HDF5 speaker/label/event row alignment and installed Tonic metadata.
- Persist complete indices/counts and content fingerprints; reject metadata,
  class-coverage, dataset, configuration and checkpoint mismatches.
- Isolate preprocessing caches and experiment outputs; preserve both LG-DTH
  local corrections and each existing model's baseline settings.
- Add ordered user-launched phases, shared OS locking, stop-on-failure behavior,
  full resume guards, distinct aggregates and existing runner cost measurement.
- Add non-training fixture tests, actual-data preflight evidence and the
  [experiment guide](shd-speaker-holdout.md). No training was launched.
