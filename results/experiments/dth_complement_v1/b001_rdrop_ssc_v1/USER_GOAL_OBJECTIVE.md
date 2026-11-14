Update the existing Goal to follow this focused research plan. Save the detailed plan under docs/experiments/ and use a short Goal referencing it. Preserve existing usage accounting.

OBJECTIVE

Improve DTH with one complementary idea B, initially on SSC, then check ECG and preservation of SHD* performance.

Existing three-seed ordinary-test means:
- SSC: DTH approximately 79.95%; published SE-adLIF 80.44%.
- ECG: DTH approximately 87.63%; published SE-adLIF 88.18%.

Verify these against authoritative records. These are means, not required outcomes for an individual seed.

Success requires improvement over matched DTH controls, not merely crossing a rounded paper number. Final paper comparisons require complete budgets and multiple seeds.

AUTHORIZATION

For this study, I authorize launching, resuming, and safely stopping its training jobs. This overrides AGENTS.md’s user-launch-only restriction. Preserve unrelated jobs and historical evidence. Do not commit or push.

Use existing training entry points and isolated opt-in configurations. Do not create another runner. New DTH+B models start from scratch; do not substitute weights-only fine-tuning for a fresh comparison.

EFFICIENCY

RTX 4090, 24 GB VRAM. No fixed wall-clock deadline. Respect the actual native Goal token budget; use 50,000 if supported and not already configured. Never reset accounting to bypass limits.

Inspect existing audits, tuning studies, failures, and compatible controls once. Avoid duplicate experiments and repeated broad repository reviews.

Prefer a small, interpretable change before an expensive redesign. There is no fixed shortlist for B. Explain its hypothesis, prior-art distinction, and why it should help DTH transfer across tasks. Do not present ordinary hyperparameter tuning as a novel scientific contribution.

Benchmark larger batch sizes briefly using disposable training states:
- SSC: consider 256, 512, 1024.
- ECG: consider 64, 128, 256.

Measure throughput, peak memory, representative long sequences, and optimizer updates per epoch. Keep about 15% memory headroom. Choose measured efficiency, not the largest fitting batch.

Freeze batch size and precision across matched arms. A batch change requires a compatible control; do not compare against an old smaller-batch result as an architecture-only improvement. Do not automatically scale LR or silently change sequence lengths, preprocessing, or backpropagation.

Reuse verified calibration when applicable. Keep one training job per GPU.

STAGE 1 — SSC DISCOVERY, SEED 42

1. Recover the existing DTH seed-42 control and verify its split, batch, precision, optimization, budget, checkpoint selection, and source compatibility.
2. Reuse it only when compatible. Otherwise explain the mismatch and train a new matched control.
3. Propose one idea B and freeze its configuration, screening rule, practical improvement threshold, and compute estimate before training.
4. Run targeted correctness checks and a short smoke check. Discard smoke states.
5. Screen fresh DTH+B for ten complete epochs against a control at the same budget.
6. Extend promising or genuinely inconclusive comparisons to 40 total epochs, preserving the intended scheduler settings.
7. Do not abandon a method solely because early learning is slow. Do not repeatedly extend an unpromising method without evidence.

Use official SSC validation. Select checkpoints using validation only.

STAGE 2 — SSC EXPLORATORY TEST CHECK

Freeze the candidate and checkpoint identities before testing.

Evaluate the selected DTH+B and matched DTH checkpoints once on official SSC test data, ordinary single-pass inference, no TTA. Reuse compatible verified control test records.

Report the actual paired seed-42 improvement and comparison with the published mean, clearly labeled as a single-seed comparison.

The user requests test-based progress checks. Any subsequent search influenced by these scores is EXPLORATORY. Record that exposure explicitly. Never describe a later winner as independently confirmed on an untouched test set. Do not search checkpoint epochs, seeds, or TTA settings using test scores.

STAGE 3 — ECG TRANSFER, SEED 42

For a shortlisted B, use the same mechanism on ECG with its inherited dataset-specific settings.

Verify ECG representation, masks, labels, and split separation. Do not claim patient independence without patient metadata.

Compare against compatible DTH controls. Use staged checks around 50 and 100 epochs. Complete the 400-epoch budget before a full-budget paper comparison. Underlearned early controls are not evidence of a large final gain.

Select checkpoints with validation and perform a frozen exploratory test check afterward. Report failure to transfer honestly.

STAGE 4 — PRESERVE SHD* PERFORMANCE

Before expanding seeds, check that B does not materially damage DTH’s existing SHD* strength.

Use canonical SHD*: 20% random-recording validation from official training, split seed 42. Keep speaker-holdout diagnostics separate.

Use short matched-budget screening, with 300 epochs required for a full-budget comparison. Never substitute checkpoints trained under another split.

STAGE 5 — FREEZE AND COMPLETE ABLATIONS

Only after cross-dataset evidence is promising, freeze B and evaluate:
- SE-adLIF.
- DTH.
- SE-adLIF+B.
- DTH+B.

Treat DTH as the existing first component. Verify whether B adds value beyond DTH and whether the combined model exceeds both individual components. Do not assume synergy.

Expand to matching seeds 123 and 456 only after this freeze. Reuse compatible existing controls. More seeds may be necessary if effects remain small or unstable.

ADAPTIVE LOOP

If a candidate is unpromising, preserve its evidence, diagnose the failure, and choose another justified B within remaining resources. Record whether the decision used validation or exposed test results.

Keep a compact registry of hypotheses, configurations, budgets, outcomes, and decisions. Do not run an unlimited queue or silently increase stage budgets.

LOGGING

Print flushed stage-start messages and one compact line after every epoch:

[stage][idea][dataset][seed][completed/ceiling]
train_acc, train_loss, val_acc, val_loss, best_epoch,
LR, batch, updates, epoch_seconds, peak_VRAM, measured ETA.

Print each decision as ADVANCE, EXTEND, REVISE, REJECT, or BLOCKED, with brief evidence and the next action.

Save resolved configs, metrics, best/last checkpoints, checkpoint identities, STATUS.json, and a concise results report. Distinguish completed epochs from zero-based indices.

Monitor approximately every ten minutes or at errors/stage boundaries. Avoid full-log dumps and repeated long analyses while waiting.

STOPPING AND REPORTING

Stop launching work before token exhaustion. Preserve progress and disclose any active bounded job.

Report test and validation separately, all failed candidates, per-seed results, mean/sample SD, paired improvements, practical effect size, costs, and protocol deviations.

Do not promise significance, superiority, two complementary contributions, or publication. Stop with an honest negative result if no justified candidate succeeds.