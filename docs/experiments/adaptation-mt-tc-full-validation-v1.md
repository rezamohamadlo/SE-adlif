# MT-TC versus MT: full SHD* validation

User execution completed 12 November 2026. Read-only checks matched the
completion identity, all 58 output hashes, all 24 reused value hashes,
prediction coverage, six saved clean-anchor tolerances and supervised-loss
metadata. See the [post-run review](../../results/experiments/adaptation_mt_tc_full_validation_v1/POST_RUN_REVIEW.md).
No evaluation rerun is needed. MT-TC-specific statistics and window-evidence
diagnostics are implemented and preflight checked, awaiting user execution;
see the [statistics/window guide](adaptation-mt-tc-statistics-and-windows-v1.md).
No inference, statistics or tests were rerun
by the agent.

The [versioned study](../../config/studies/adaptation_mt_tc_full_validation_v1.yaml)
inherits the full-validation definitions and `config/paper_baseline.yaml`
authority. It pins existing MT checkpoints (epochs 199/137/107 for seeds
42/123/456) and MT-TC checkpoints (232/249/226), authenticating exact bytes,
selected epochs, saved configs and original metrics. Historical checkpoint
paths resolve through registered aliases without changing file identity.

## Training objective versus comparison loss

Both models use the same MT neuron architecture, common optimizer/settings,
20% SHD* validation split with split seed 42, 300-epoch budget, batch size 256,
LR .01 and dropout .15. These values are checked against the inherited paper
baseline rather than modified for this evaluation. MT-TC intentionally has a
different training objective: `mt_temporal_consistency_v1` with JS weight .1
and window 5. Its distinct saved protocol identity is retained. Shared baseline
settings do not make the training objectives identical.

The inference model uses the checkpoint's strict-loaded ordinary `MLPSNN`
forward; the MT-TC training wrapper changes the training step but not parameter
keys or forward dynamics. No training factory, optimizer, training step or
auxiliary JS function is invoked by this evaluation. The paired benefit is
**supervised cross-entropy(MT) minus supervised cross-entropy(MT-TC)**. It never
includes JS or the logged total training loss. Saved `val_loss` is also
supervised because the auxiliary penalty is implemented only in the training
step. Report metadata preserves the intentional objective difference.

The hypothesis concerns stable class evidence across neighboring windows.
The existing fast/slow contrast and input ACF are not direct measures of that
evidence; this descriptive evaluation must not be presented as testing the
complete temporal-consistency mechanism. A dedicated statistical specification
and any new window-level diagnostic definitions are separate future work.

## Launch locally

From the repository root using its installed `.venv`:

```powershell
# Optional preflight: no forward passes, result writes or tests.
.\.venv\Scripts\python.exe scripts/adaptation_mt_tc/evaluate.py --check-only

# USER LAUNCH: full frozen MT-TC evaluation plus authenticated MT reuse.
.\.venv\Scripts\python.exe scripts/adaptation_mt_tc/evaluate.py

# USER LAUNCH after completion: verify the self-contained output package.
.\.venv\Scripts\python.exe scripts/adaptation_mt_tc/evaluate.py --verify-only
```

The same commands are in [command.txt](../../command.txt). This is a short
single-machine evaluation; output identities do not depend on PC assignments.
The script resolves paths against its own checkout even when the working
directory changes. Preflight verifies strict state loading without inference.
No training, new test-set access, TTA or downloads occur.

All 1,631 saved validation members, training seeds 42/123/456 and six conditions/
replicates are retained: clean, time scales .9/1.1, and event deletion .1 with
three realizations. The original completed MT/AMT full-validation package is
pinned as the reuse source. Checkpoints/configs/metrics, source implementations,
environment, samples, split, raw data, features and perturbation definitions
must match before reusing the eighteen MT prediction chunks and six feature
chunks. Any mismatch fails preflight; there is no silent fallback to different
weights, settings or a new parent evaluation.

During user execution, derived caches retain original value hashes and record
original identities/paths/byte hashes in `REUSE_PROVENANCE.json`. Only MT-TC's
eighteen prediction chunks require new inference. Every transformed input is
checked against the shared cache, and full clean anchors must match saved
validation evidence. Reused predictions and perturbation realizations do not
add independent training seeds. Reports retain per-seed values, means and
sample SD; deletion realizations are averaged inside each seed for the
additional descriptive effect table.

Outputs use `results/experiments/adaptation_mt_tc_full_validation_v1/`, with
manifest, reuse provenance, clean anchors, progress and completion records,
shared `diagnostic_rows/` and `analyses/clean`, `analyses/time_scaling`,
`analyses/spike_deletion`. Terminal output and `PROGRESS.json` show progress.
Only one worker may write the directory. Interrupted runs reuse complete,
identity-checked chunks; this is evaluation continuation, not training resume.
Changed settings, code or environment require a new output root.

Share the COMPLETE result directory, including ignored `diagnostic_rows/`,
with the same repository-relative layout. Git alone does not transfer all
records. Completed-package verification needs no checkpoint, old reuse source,
dataset or GPU; evaluation continuation still needs its original prerequisites.

Prepared tests, not executed by the agent:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_mt_tc tests.test_adaptation_full_validation
```

After user execution, inspect the saved evidence and update the roadmap before
defining MT-TC-specific adjusted statistics or window-evidence diagnostics.
No subsequent stage starts automatically.
