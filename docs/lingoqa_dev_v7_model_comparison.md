# LingoQA Dev V7 Model Comparison

Baseline: `sft_v2`
Incumbent: `v5c_step20`

| run | status | normal | text | wrong | blank | setting_gap | case_gap | pos_rate | d_normal | d_case_gap | d_normal_inc | d_setting_inc | d_case_inc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft_v2 | baseline | 0.3680 | 0.2873 | 0.2965 | 0.2564 | 0.0715 | -0.0070 | 0.2200 | 0.0000 | 0.0000 | -0.0147 | -0.0130 | -0.0121 |
| v5c_step20 | incumbent | 0.3827 | 0.2914 | 0.2982 | 0.2826 | 0.0845 | 0.0051 | 0.2700 | 0.0147 | 0.0121 | 0.0000 | 0.0000 | 0.0000 |
| v6_step15 | reject_incumbent_regression | 0.3732 | 0.2721 | 0.2987 | 0.2697 | 0.0745 | 0.0106 | 0.2400 | 0.0052 | 0.0176 | -0.0095 | -0.0100 | 0.0055 |
| v7_step12 | reject_incumbent_regression | 0.3696 | 0.2718 | 0.2964 | 0.2585 | 0.0732 | 0.0048 | 0.2500 | 0.0016 | 0.0118 | -0.0131 | -0.0113 | -0.0003 |
| v7_step24 | reject_incumbent_regression | 0.3763 | 0.2693 | 0.2883 | 0.2686 | 0.0880 | 0.0091 | 0.2400 | 0.0083 | 0.0161 | -0.0064 | 0.0035 | 0.0040 |

Decision rule used by this report:

- `baseline`: reference run for deltas.
- `incumbent`: current best local run that new checkpoints should beat.
- `candidate`: normal F1 and visual gaps pass the configured floors.
- `diagnostic_only_gap_weak`: normal F1 is acceptable, but per-case visual dependency is still weak.
- `reject_normal_regression`: normal-image quality regressed below the configured floor.
- `reject_incumbent_regression`: acceptable versus baseline, but weaker than the incumbent on normal F1 or setting-level visual gap.
