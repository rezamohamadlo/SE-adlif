# User-requested DTH + R-Drop continuations: SHD100 / SSC70 / ECG70

Continue seed42 from each experiment's own latest complete state, including
weights, Adam moments, scheduler, callback selection and epoch/step counters.
Targets are total completed epochs, not additional epochs. Starts: SHD30,
SSC31, ECG30. Only DTH + R-Drop continues; historical DTH controls stay30.

Use the official runner with the explicit extended study definition. Preserve
all old directories, checkpoints, tests and the completed30comparison report.
The new output root is `results/experiments/dth_complement_v1/extended_100_70_70_v1/`.

SHD uses its unchanged canonical SHD* random20percent validation, splitseed42,
batch256 and original300epoch schedule. ECG retains row-wise5percent
validation, splitseed42, signed inputs/masks, batch64 and original400schedule.
SSC retains its official validation, batch256, valbatch1024, workers8 and
spawn-safe preprocessing. All keep dropout0.15, alpha1, temperature1,
epsilon0, original LR/loss/neurons and constant ReduceLROnPlateau settings.

SSC70 exceeds the paper baseline40. It is a distinct, explicit budget
deviation, not a new baseline version or a matched paper reproduction.
An opt-in SSC-only guard permits this extension only with the original40
schedule identity, plateau scheduler and unchanged high patience9999. No
cosine/horizon-dependent schedule may be silently extended by this policy.
The paper baseline file, run.py, runner.py and model/data implementations
remain unchanged.

Prepare each new seed directory by copying its own history and checkpoint
state. A derived checkpoint changes only saved configuration/routing and
callback paths, with every difference recorded. Do not reset epoch, global
step, optimizer or scheduler. Verify recursive hashes of model, optimizer,
scheduler and loop state against the immutable original before training.
Keep original checkpoint SHA256s, configs, initial-state references and launch
provenance in per-run continuation records. The derived file has its own hash;
never claim it has the original file identity. Resume from its own new last.

Run one GPU job at a time: SHD100, ECG70, then SSC70, returning early results
before the long SSC continuation. Freeze each validation-selected checkpoint
after its target completes, then evaluate ordinary official-test inference
without TTA. No epoch/seed selection using test scores and no score-based
changes to later jobs. Prior tests were exposed; all results are exploratory.

Update a separate extended-budget table after each completed test. Label
historical DTH controls as30epochs and therefore budget-unmatched with the
extended model. Preserve the matched30table separately. Report baseline budget
differences, per-seed values, selected epoch/hash and signed paper deltas.
ECG accuracy is per valid time-step under the original masks; patient
independence is not established. Stop after the three requested targets/tests.
