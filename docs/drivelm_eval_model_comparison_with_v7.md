# DriveLM Visual-Control Model Comparison With V7

Baseline: `base`

| run | status | normal | text | wrong | blank | setting_gap | case_gap | pos_rate | d_normal | d_case_gap |
|---|---|---|---|---|---|---|---|---|---|---|
| base | baseline | 0.2910 | 0.2826 | 0.2907 | 0.2820 | 0.0003 | -0.0970 | 0.0000 | 0.0000 | 0.0000 |
| sft_v2 | diagnostic_only_setting_gap | 0.2871 | 0.3052 | 0.2870 | 0.3203 | -0.0332 | -0.1067 | 0.0067 | -0.0039 | -0.0097 |
| v5c_step20 | diagnostic_only_setting_gap | 0.2824 | 0.3139 | 0.2822 | 0.3231 | -0.0407 | -0.1205 | 0.0067 | -0.0086 | -0.0235 |
| v7_step24 | diagnostic_only_setting_gap | 0.2816 | 0.3065 | 0.2816 | 0.3156 | -0.0340 | -0.1198 | 0.0033 | -0.0094 | -0.0228 |

Decision rule used by this report:

- `baseline`: reference run for deltas.
- `incumbent`: current best local run that new checkpoints should beat.
- `candidate`: normal F1 and visual gaps pass the configured floors.
- `diagnostic_only_gap_weak`: normal F1 is acceptable, but per-case visual dependency is still weak.
- `reject_normal_regression`: normal-image quality regressed below the configured floor.
- `reject_incumbent_regression`: acceptable versus baseline, but weaker than the incumbent on normal F1 or setting-level visual gap.
