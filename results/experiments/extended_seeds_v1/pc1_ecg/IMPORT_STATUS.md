# Imported ECG extension snapshot

Imported on 29 October 2026 from the user-supplied external snapshot `F:/reza/pc1_ecg_9seeds`.

- MT: seven completed 400-epoch training/validation runs, seeds 1009, 2027, 3037, 4049, 5051, 6067, 7079. Each already contains a standalone test result.
- SA-MT: seed 1009 completed 400 epochs; seed 2027 completed 38 epochs, with a retained last checkpoint and optimizer state. No SA-MT test records were provided.
- All 130 copied files were SHA-256 verified; saved configs, metrics, validation selection and checkpoint metadata were checked. Source evidence was preserved byte for byte.
- The strict validation-only aggregator rejects the MT test artifacts. Preserve this distinction; do not remove test evidence or claim these are test-free records.
- Source launch commit is `b5ae11a`; resume with the current checkout has not been validated. No training or test inference was launched, and no current comparison report was updated.

[Transfer manifest](../../../provenance/ecg-extension-import-2026-10-29.json)
