# Speaker-holdout regularization: official test

User-requested test of six completed candidates; no test-based selection. Incomplete tuning grid; not the random-20%-validation SHD* protocol.

Ordinary inference, no TTA. Best checkpoint chosen using validation only.

| Profile | Model | Seed | Validation (%) | Test (%) |
|---|---|---:|---:|---:|
| dropout035_v1 | DTH_SE_adLIF | 456 | 90.4587 | 91.6078 |
| dropout035_v1 | LG_DTH_SE_adLIF | 456 | 91.2538 | 92.7120 |
| spike_delete010_v1 | DTH_SE_adLIF | 123 | 91.3761 | 93.6837 |
| spike_delete010_v1 | LG_DTH_SE_adLIF | 123 | 90.4587 | 94.1254 |
| spike_delete010_v1 | DTH_SE_adLIF | 456 | 92.5382 | 94.4788 |
| spike_delete010_v1 | LG_DTH_SE_adLIF | 456 | 92.3547 | 93.1537 |

| Profile | Model | n | Test mean ± sample SD (%) |
|---|---|---:|---:|
| dropout035_v1 | DTH_SE_adLIF | 1 | 91.6078 ± SD unavailable |
| dropout035_v1 | LG_DTH_SE_adLIF | 1 | 92.7120 ± SD unavailable |
| spike_delete010_v1 | DTH_SE_adLIF | 2 | 94.0813 ± 0.5622 |
| spike_delete010_v1 | LG_DTH_SE_adLIF | 2 | 93.6396 ± 0.6871 |
