# TTA result import, 26 October 2026

Fast-forwarded main from fb9b02a to remote commit 8871ac1. Preserved the
unrelated local deletion of prmmt.txt. Verified and copied/matched 227 files
from the TTA experiment directories under external source F:/reza/results/experiments.
The external source remains unchanged. No prior local files differed.

Verified all 72 four-candidate evaluation records (36 validation, 36 test):
manifest fingerprints, exact local checkpoint SHA-256 identity, dataset/seed/split,
fixed gain policy and reported accuracy differences. Dataset fingerprints match
across all four candidate models for each dataset and evaluation split.
Copied historical MT/DTH/MT-DTH TTA directories too, without altering their values.

The shared manifest is results/provenance/reza-tta-import-2026-10-26.json.
The comparison summary now includes all four candidates' TTA test means and
sample SDs, with paper differences and exploratory-evaluation caveats.
The remote launch command's .ven typo was corrected to the project's .venv.
No training or evaluation was launched during import. No commit or push was made.
