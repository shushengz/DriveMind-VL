# DriveLM Visual-Preference V2 Comparison

Baseline: `base`
Incumbent: `sft_1200`

| run | status | normal | text | wrong | blank | setting_gap | case_gap | pos_rate | d_normal | d_case_gap | d_normal_inc | d_setting_inc | d_case_inc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | baseline | 0.2615 | 0.2689 | 0.2622 | 0.2584 | -0.0074 | -0.0924 | 0.0633 | 0.0000 | 0.0000 | -0.1167 | 0.0329 | 0.0652 |
| sft_1200 | incumbent | 0.3782 | 0.3935 | 0.3627 | 0.4185 | -0.0403 | -0.1576 | 0.0900 | 0.1167 | -0.0652 | 0.0000 | 0.0000 | 0.0000 |
| visual_pref_v1 | diagnostic_only_gap_weak | 0.3789 | 0.3936 | 0.3654 | 0.4146 | -0.0357 | -0.1594 | 0.0767 | 0.1174 | -0.0670 | 0.0007 | 0.0046 | -0.0018 |
| visual_pref_v2 | diagnostic_only_gap_weak | 0.3758 | 0.3825 | 0.3731 | 0.4357 | -0.0599 | -0.1601 | 0.0667 | 0.1143 | -0.0677 | -0.0024 | -0.0196 | -0.0025 |

Decision rule used by this report:

- `baseline`: reference run for deltas.
- `incumbent`: current best local run that new checkpoints should beat.
- `candidate`: normal F1 and visual gaps pass the configured floors.
- `diagnostic_only_gap_weak`: normal F1 is acceptable, but per-case visual dependency is still weak.
- `reject_normal_regression`: normal-image quality regressed below the configured floor.
- `reject_incumbent_regression`: acceptable versus baseline, but weaker than the incumbent on normal F1 or setting-level visual gap.
