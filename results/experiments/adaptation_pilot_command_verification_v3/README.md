# Analysis command verification

Technical checks on 2026-11-10 using the local `.ven` Python and RTX 4090.
These outputs verify the launcher; they are not the planned scientific cohort.
The default `adaptation_mechanism_analysis_v1` experiment was not started.

| Command arm | Original training seed | Pilot validation recordings | Clean validation anchor per checkpoint |
|---|---:|---:|---:|
| PC1, feature association | 42 | 12 | 1,631 |
| PC2, time scaling | 123 | 12 | 1,631 |
| PC3, spike deletion | 456 | 12 | 1,631 |

Each arm completed both MT and AMT inference and reporting. All six full-clean
accuracy/loss anchors matched their selected-epoch CSV evidence within the
frozen tolerances. Conditions were evaluated separately; no training, optimizer
steps, official-test access or TTA occurred. The three arms here use different
seeds for technical coverage and must not be treated as a matched three-seed
mechanistic comparison. The user's default commands run all three seeds per arm.

Recorded analysis times, excluding prerequisite checks, were 13.94, 15.52 and
16.73 seconds for these small PC1/PC2/PC3 checks. They are not training times or
performance comparisons between computers.

The 49 targeted feature, data, reporting, CLI, pairing and manifest tests passed.
Default-cohort preflight passed for all three roles. The wrapper also passed
preflight from another working directory. Identical reruns verified completion
hashes and skipped; changed settings were rejected. The local `.venv` executable
does not exist, so the PC1 command's Python program was tested with `.ven`;
`command.txt` explicitly uses the requested `.venv` name for PC1.

`MANIFEST.json`, clean anchors and summaries are beside each arm. Detailed
`diagnostic_rows/` are ignored local evidence; transfer those too to continue or
fully reverify an existing output folder on another PC. Earlier v1/v2 technical
directories retain their original source identities and are superseded by this
verification. They are not additional training runs.
