# DriveLM Scene-Heldout Visual-Control Comparison

Baseline: `base`

| run | status | normal | text | wrong | blank | setting_gap | case_gap | pos_rate | d_normal | d_case_gap |
|---|---|---|---|---|---|---|---|---|---|---|
| base | baseline | 0.2615 | 0.2689 | 0.2622 | 0.2584 | -0.0074 | -0.0924 | 0.0633 | 0.0000 | 0.0000 |
| drivelm_sft | diagnostic_only_gap_weak | 0.3782 | 0.3935 | 0.3627 | 0.4185 | -0.0403 | -0.1576 | 0.0900 | 0.1167 | -0.0652 |

Decision rule used by this report:

- `baseline`: reference run for deltas.
- `incumbent`: current best local run that new checkpoints should beat.
- `candidate`: normal F1 and visual gaps pass the configured floors.
- `diagnostic_only_gap_weak`: normal F1 is acceptable, but per-case visual dependency is still weak.
- `reject_normal_regression`: normal-image quality regressed below the configured floor.
- `reject_incumbent_regression`: acceptable versus baseline, but weaker than the incumbent on normal F1 or setting-level visual gap.
