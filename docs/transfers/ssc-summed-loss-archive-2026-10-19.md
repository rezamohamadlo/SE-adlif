# SSC summed-loss archive - 19 October 2026

At user request, all three completed seeds for the reference, DTH and MT were moved out of active `results/valid/` SSC directories into `archive/ssc_summed_loss_20261019/<model>/SSC/`. All 123 original files were checked by SHA256 after moving. Pre-archive main/model reports were retained as snapshots inside that archive. These remain evidence for the original summed-loss protocol; archiving is not a finding of invalid training.

The [manifest](../../results/provenance/ssc-summed-loss-archive-2026-10-19.json) records source/destination paths and all file hashes. Active model tables show SSC as pending, and main-report SSC historical scores are explicitly labeled. Links to moved evidence were updated.

The separate mean-loss reference diagnostic was preserved in place, including its seed-42 best validation accuracy of 82.22%. It has not been promoted into the canonical reference. No official test evaluation was inferred, no process was stopped or started, and default baseline/config values were not changed. The user manages the ongoing diagnostic process separately.

Future mean-loss reference/DTH/MT comparisons require explicit matched configs and separate protocol metadata. Empty active directories alone do not change the runner's loss selection.
