# IntelliCockpitBench Ablation Breakdown Report

This report extends the 7-sample IntelliCockpitBench smoke test with capability-level external VQA metrics and per-case visual dependency gaps.

## Why This Matters

The previous ablation compared average `external_answer_f1` across settings. That is useful, but not strict enough: a model may do better on the normal-image average while individual samples still have better text-only, wrong-image, or blank-image answers.

This report therefore adds:

- external VQA breakdown by capability, category, subcategory, shooting angle, and weather;
- per-case `visual_dependency_gap`;
- capability-level visual dependency summaries.

## Commands

After running `scripts/18_run_intelli_visual_ablation.sh`, run:

```bash
bash scripts/19_analyze_intelli_ablation_breakdown.sh
```

## Normal-Image Capability Breakdown

| Capability | Count | External Answer F1 | Pass Rate | Avg Reward |
|---|---:|---:|---:|---:|
| counting | 2 | 0.2632 | 0.5000 | 0.4540 |
| object_recognition | 1 | 0.0000 | 0.0000 | 0.3750 |
| spatial_localization | 3 | 0.1618 | 0.3333 | 0.4235 |
| weather_road_condition | 1 | 0.4375 | 1.0000 | 0.5062 |

## Per-Case Visual Dependency

`visual_dependency_gap = normal_f1 - max(text_only_f1, wrong_image_f1, blank_image_f1)`

| Group | Count | Normal F1 | Per-Case Control Max F1 | Visual Gap |
|---|---:|---:|---:|---:|
| overall | 7 | 0.2070 | 0.2278 | -0.0207 |
| counting | 2 | 0.2631 | 0.0000 | +0.2631 |
| object_recognition | 1 | 0.0000 | 0.6667 | -0.6667 |
| spatial_localization | 3 | 0.1618 | 0.1981 | -0.0363 |
| weather_road_condition | 1 | 0.4375 | 0.3333 | +0.1042 |

## Interpretation

This is the strictest result so far. The normal-image average is better than text-only and wrong-image averages, but per-case visual dependency is not consistently positive. On this small sample, original images help counting and weather/road condition, but do not help object recognition or spatial/scene-completeness reliably.

The object recognition case is especially important: the model predicts `toyota` while the reference is `Volkswagen Passat`; one control condition overlaps more with the reference by chance. This shows why a serious report cannot rely only on total averages.

## Current Claim Boundary

A rigorous claim is:

> Qwen2.5-VL-3B follows the DriveMind external VQA schema reliably, but the 7-sample IntelliCockpitBench smoke test does not provide sufficient evidence of robust visual grounding. Per-case ablation shows mixed image dependence, with positive gaps in counting and weather/road condition but failures in object recognition and scene/spatial detail.

## Next Step

Use the same scripts on 50-100 IntelliCockpitBench samples. The next milestone should report:

- capability-level normal metrics;
- capability-level visual dependency gaps;
- per-case worst failures;
- manual review for at least 20 sampled cases.
