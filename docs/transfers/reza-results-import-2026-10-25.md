# Import of LT-DTH and MT-TC seed-42 results

On 25 October 2026, copied 12 completed runs from the explicitly supplied
external directory `F:/reza`: nine LT-DTH runs (SHD*/SSC/ECG, seeds 42/123/456)
and three MT-TC runs (SHD*/SSC/ECG, seed 42). Source copies were retained.

Destinations:

- LT-DTH: `results/not_improved/LT_DTH_SE_adLIF/<dataset>/seed_<seed>/`.
  This retention category follows its SHD* outcome, not a cross-dataset verdict.
- MT-TC: `results/experiments/mt_consistency_runner_v1/MT_TC_SE_adLIF/<dataset>/seed_42/`.
  Local seeds 123/456 were not overwritten, stopped or restarted.

The transfer copied 170 files totaling 208,747,288 bytes, including hidden
Hydra configs, metrics, completion markers, all checkpoints and cost evidence.
Each copied file was SHA256-verified against its source. All 12 best-checkpoint
hashes also match their saved `COST_RESULTS.json` identities. Validation histories
cover the full budgets with no duplicate validation epochs; final test metrics
agree with reproduced cost-evaluation accuracies within floating point tolerance.

The source MT-TC root was named `mt_tc_v1`; the active local root is named
`mt_consistency_runner_v1`. Explicit aliases in `config/path_aliases.yaml`
normalize this root difference and relocate LT-DTH's original `lt_dth_v1` root
to the retention area. Saved configs, checkpoint bytes and cost JSONs were not
rewritten. The model/dataset/seed/checkpoint suffixes are preserved exactly.

The LT-DTH source aggregate remains at its retained model root. The source
MT-TC seed-42-only aggregate was preserved under the local model's
`transfer_history/2026-10-25-seed42/RESULTS.md`, preserving the active local
aggregate instead of overwriting it with an incomplete three-seed view.
The active runner will aggregate imported completions alongside local runs
as it progresses. No second launcher or training process was started.

Non-training checks recognized all three imported MT-TC seed-42 runs and all
nine LT-DTH runs as `complete`, with matching resolved configurations. No
retraining, test reevaluation or cost reevaluation was needed for the import.

[Shared per-file transfer manifest](../../results/provenance/reza-results-import-2026-10-25.json)
records source locations, repository-relative destinations, file sizes, hashes,
run identities, baseline/protocol metadata and checkpoint identities.
