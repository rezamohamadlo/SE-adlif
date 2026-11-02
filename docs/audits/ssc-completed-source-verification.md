# SSC completed-source verification

On 2026-11-02, LR selection was blocked by the resume provenance guard.
All 18 completed LR runs recorded commit `d42804ad8e5ff0761c626d83585440412f14efea`
and source fingerprint `8778d1d9e336c3b9a7306219af3eab0dcc74c7438eef2d0fd8dde48381b81303`.
Their broad tracked-diff fingerprints differed because PC3's archived checkout
also contained a modification to `reports/comparison-summary.md`.

Completed SSC evidence now checks internal provenance consistency and rebuilds
the source fingerprint from each recorded Git commit. A missing revision or a
source fingerprint differing from that revision fails selection. This deliberately
does not automatically accept historical source modifications absent from Git.
Current checkout changes no longer masquerade as changes to completed training.
Training/resume provenance guards remain unchanged, as do completed records.

The official `runner.py --ssc-study-stage select-lr` operation succeeded.
Both `dropout` assignments passed `--dry-run`: SA-MT at LR 0.009 and AMT at
LR 0.003, dropout 0.10/0.20, seed 42 on PC3 and 123 on PC4. No training or
test evaluation was launched. Future launches must use the main checkout with
the updated study helper and frozen selection, rather than the older archived
helper. Transfer the updated helper and selection to PC4 before its launch.
