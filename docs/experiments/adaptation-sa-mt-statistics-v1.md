# SA-MT versus MT: adjusted validation statistics

User execution completed 12 November 2026. Read-only checks matched the
completion identity and all eight output hashes. The saved summary reports
1,631 complete cases and 2,000 valid bootstrap draws. See the
[summary](../../results/experiments/adaptation_sa_mt_full_validation_v1/statistics_v1/SUMMARY.md)
and [interpretation](../../results/experiments/adaptation_sa_mt_full_validation_v1/statistics_v1/INTERPRETATION.md).
No rerun is needed. The agent ran only preflight before launch and inspected
saved files afterward; no analysis or tests were rerun by the agent.

The [versioned definition](../../config/studies/adaptation_sa_mt_statistics_v1.yaml)
pins the exact completed SA-MT/MT manifest and completion hashes. The source
contains 1,631 validation samples, training seeds 42/123/456 and the original
MT predictions reused in the SA-MT evaluation. Reuse provenance and unchanged
parent/input value hashes are checked within this self-contained source package;
the earlier AMT package is not needed to perform these statistics.

The hypothesis is a positive adjusted association between SA-MT loss benefit
and the frozen fast/slow input-activity contrast as a **proxy** for transient
versus sustained input activity. This is not direct evidence of branch-specific
adaptation strengths or mediation. Autocorrelation crossing lag remains a
two-sided exploratory covariate. Both features enter the same model, forming
the two-feature multiplicity family. No features, subgroup thresholds or
checkpoints are chosen to make the previously observed results positive.

The response is `loss(MT) - loss(SA-MT)`, positive favoring SA-MT. Numerical
methods are inherited from the established [statistics specification](adaptation-statistics-v1.md):
class and speaker fixed effects, log1p(valid frame duration), log1p(total input
activity), fixed full-cohort scaling, explicit complete-case exclusions,
seed-specific estimates and leave-one-speaker-out sensitivity. MT loss is
not used as a difficulty control because it is part of the response itself;
unmeasured sample difficulty remains a possible confounder.

Uncertainty uses 2,000 whole-speaker bootstrap draws shared across all three
training seeds and intervention outcomes. Nominal 97.5% individual intervals
adjust the two-feature family; a separate three-intervention family uses
nominal 98.3333% intervals. These are approximate sensitivity summaries with
only ten speakers, conditional on the three observed model pairs, not reliable
formal significance tests or uncertainty over a population of training seeds.
Intervals for individual seeds are diagnostic rather than a six-test family.
There are no p-values or claims of causal/independent test confirmation.

Secondary effects are within-sample changes in relative benefit for time
scales .9 and 1.1 and 10% deletion. The three deletion realizations are averaged
within each sample and training seed first. Reused MT predictions, repeated
samples and perturbation realizations do not increase the number of training
seeds or create independent replications. Negative and inconclusive findings
must remain in the output.

The specification was prepared after descriptive validation results were
observed, so it is exploratory and not preregistration. No neural weights or
branch-strength parameters are loaded, and no new inference, training,
official-test access or data download occurs. A separate hypothesis/source
identity is used for this model pair; previous AMT/LT-DTH results are preserved.

Run from the checkout root using its `.venv`:

```powershell
# Optional preflight: reads/verifies saved files, constructs the design;
# no adjusted fits, resampling, inference, tests or result-directory writes.
.\.venv\Scripts\python.exe scripts/adaptation_sa_statistics/analyze.py --check-only

# USER LAUNCH: adjusted regression, bootstrap and figures.
.\.venv\Scripts\python.exe scripts/adaptation_sa_statistics/analyze.py

# USER LAUNCH after completion: verify saved output hashes.
.\.venv\Scripts\python.exe scripts/adaptation_sa_statistics/analyze.py --verify-only
```

Commands are also in [command.txt](../../command.txt). This is short local CPU
analysis; no PC-count question or distributed workflow is needed. Paths resolve
from the checkout root, independently of the current/Hydra working directory.
The script writes to
`results/experiments/adaptation_sa_mt_full_validation_v1/statistics_v1/`,
with manifest, summary, exclusions, PNG/SVG figures, detailed sample CSV,
bootstrap estimates and completion hashes. Share the complete folder including
ignored `diagnostic_rows/`; Git alone does not transfer all evidence.

An identical completed invocation verifies and skips. Changed implementation,
specification or environment requires a new output root; guards are not removed.
Do not overwrite completed evaluation or statistical records.

Prepared tests, not executed by the agent:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_adaptation_sa_statistics tests.test_adaptation_statistics
```

After user execution, inspect saved outputs and update the roadmap before
preparing MT-TC/MT or additional datasets. No later stage starts automatically.
