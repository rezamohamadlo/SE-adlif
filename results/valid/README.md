# Results using separate validation and test sets

- `protocol_v2/`: controlled model runs and their resumable artifacts. A run
  is a completed result only after `completed.txt` and final test metrics exist.
- `reference/SHD/`: retained reference runs, training seeds 42, 123, 456,
  using 20% training-derived validation with split seed 42.
- `reference/SSC/`: retained reference runs using official validation/test splits.

New runner jobs write to `results/valid/protocol_v2`. Existing jobs resume from
their relocated last checkpoints; saved callback paths were updated during
the move. Best checkpoint weights remain unchanged.

Historical test-selected SHD runs and reference ECG runs with unmatched early
stopping are retained under `results/archive` and excluded from the comparison.
The old ECG scores still measure an independent test set, but they do not use
the same stopping protocol as the candidates.
