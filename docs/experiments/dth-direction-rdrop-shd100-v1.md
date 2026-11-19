# Direction-aware DTH with fixed R-Drop: bounded SHD* study

User explicitly authorizes this study's training and evaluation. No other jobs,
new ideas, seeds, tuning, commits or pushes are authorized.

A is the archived seed42 DTH+R-Drop 100-epoch run under
`results/experiments/dth_complement_v1/extended_100_70_70_v1/`.
B starts fresh with random seed42 and a fresh optimizer; it never loads A's
trained weights. B stops at100 completed epochs using the original300-epoch
paper-baseline schedule. Both use batch256, validation batch256, workers0,
dropout0.15, LR0.01, R-Drop alpha1, no label smoothing, and the same canonical
SHD* random20% split from official training, splitseed42. This is not a
speaker-disjoint study. No TTA.

The only computation change is the opt-in transient threshold response:
`-transient_gain*relu(fast-slow)+offset_gain*relu(slow-fast)`.
The bounded sigmoid mapping, original gain constraints, and other neuron
dynamics remain unchanged. Each hidden layer adds one scalar offset_gain,
initialized0, clamped[-5,5]. offset_gain=-transient_gain recovers the original
response. The hypothesis is unverified; no novelty or improvement is asserted.

Audit shared initial tensor values and RNG without optimizer updates before
launch. Freeze resolved settings, source identities and train/validation sample
indices before training. Select B's best validation checkpoint within100 epochs
before opening its test results. Archive raw predictions and per-speaker metrics.
Reuse A only as an audited historical control. A has interrupted/full-resume
history, while B is a fresh job; identical stochastic trajectories or strict
uninterrupted paired equivalence cannot be claimed. Official tests were exposed
previously, so this is exploratory and single-seed, not independent confirmation.

Training uses existing runner.py and its staged-study lock/resume safeguards.
Evaluation helpers perform no training. Resume, if necessary, only from B's own
last.ckpt, with unchanged manifests/source. Preserve best/last/optimizer states.
After the one final ordinary test, write COMPARISON_FA.md and stop. No follow-ups.
