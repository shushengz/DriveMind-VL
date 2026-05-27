# LingoQA Clean Dev Visual-Control Run Comparison

Baseline: `base_clean_dev`
Incumbent: `sft_v2_visual_scale`

| run | status | normal | text | wrong | blank | setting_gap | case_gap | pos_rate | d_normal | d_case_gap | d_normal_inc | d_setting_inc | d_case_inc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base_clean_dev | baseline | 0.2763 | 0.1902 | 0.2039 | 0.2100 | 0.0663 | -0.0484 | 0.2400 | 0.0000 | 0.0000 | -0.0917 | -0.0052 | -0.0414 |
| sft_v2_visual_scale | incumbent | 0.3680 | 0.2873 | 0.2965 | 0.2564 | 0.0715 | -0.0070 | 0.2200 | 0.0917 | 0.0414 | 0.0000 | 0.0000 | 0.0000 |
| pref_v4_guarded_simpo | reject_normal_regression | 0.0668 | 0.0510 | 0.0668 | 0.0524 | 0.0000 | -0.0040 | 0.0000 | -0.2095 | 0.0444 | -0.3012 | -0.0715 | 0.0030 |
| pref_v5_sweep_best | reject_incumbent_regression | 0.3663 | 0.2947 | 0.2868 | 0.2798 | 0.0716 | 0.0003 | 0.2400 | 0.0900 | 0.0487 | -0.0017 | 0.0001 | 0.0073 |

Decision rule used by this report:

- `baseline`: reference run for deltas.
- `incumbent`: current best local run that new checkpoints should beat.
- `candidate`: normal F1 and visual gaps pass the configured floors.
- `diagnostic_only_gap_weak`: normal F1 is acceptable, but per-case visual dependency is still weak.
- `reject_normal_regression`: normal-image quality regressed below the configured floor.
- `reject_incumbent_regression`: acceptable versus baseline, but weaker than the incumbent on normal F1 or setting-level visual gap.
