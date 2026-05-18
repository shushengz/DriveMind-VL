# LingoQA Best Strategy Visual-Control Report

## Setup

- Dataset: 100 LingoQA evaluation questions.
- Model: Qwen2.5-VL-3B-Instruct.
- Training: none.
- Strategy: `prompt_variant=spatial`, `frame_strategy=first_middle_last`, `max_images=3`.
- Controls:
  - text-only;
  - normal 3-frame;
  - wrong 3-frame;
  - blank 3-frame.

## Setting-Level Results

| setting | JSON Validity | Answer F1 | Avg Reward |
|---|---:|---:|---:|
| normal 3-frame | 1.0000 | 0.2912 | 0.4664 |
| text-only | 1.0000 | 0.1448 | 0.4256 |
| wrong 3-frame | 1.0000 | 0.2095 | 0.4411 |
| blank 3-frame | 1.0000 | 0.1349 | 0.4232 |

Setting-level visual dependency:

```text
normal_f1 - max(text_only_f1, wrong_image_f1, blank_image_f1) = 0.0817
```

This is stronger than the previous default 5-frame control result, where the setting-level gap was positive but the per-case conservative gap remained negative.

## Per-Case Conservative Gap

The stricter per-case metric compares each normal prediction against the best of its three controls:

```text
gap = normal_f1 - max(text_only_f1, wrong_frame_f1, blank_frame_f1)
```

Overall:

```json
{
  "count": 100,
  "normal_f1": 0.2912,
  "text_only_f1": 0.1448,
  "wrong_image_f1": 0.2095,
  "blank_image_f1": 0.1349,
  "control_max_f1": 0.2700,
  "visual_dependency_gap": 0.0212,
  "positive_gap_rate": 0.3100
}
```

By capability:

| Capability | Count | Normal F1 | Control Max F1 | Conservative Gap |
|---|---:|---:|---:|---:|
| counting | 21 | 0.1738 | 0.0836 | +0.0902 |
| object_recognition | 20 | 0.3791 | 0.2863 | +0.0927 |
| other | 20 | 0.4031 | 0.4180 | -0.0149 |
| reasoning_world_knowledge | 17 | 0.2034 | 0.2587 | -0.0553 |
| spatial_localization | 20 | 0.3134 | 0.3329 | -0.0195 |
| weather_road_condition | 2 | 0.0513 | 0.0513 | 0.0000 |

## Interpretation

The best strategy produces a meaningful improvement over the prior current 5-frame baseline:

| Strategy | Answer F1 | Pass Rate | Spatial F1 | Reasoning F1 |
|---|---:|---:|---:|---:|
| current + first 5 frames | 0.2508 | 0.4100 | 0.2653 | 0.1690 |
| spatial + first/middle/last 3 frames | 0.2912 | 0.5500 | 0.3134 | 0.2034 |

The improvement is not only an overall metric artifact. Under strict controls, normal 3-frame beats text-only, wrong-frame, and blank-frame at the setting level, and the conservative per-case visual dependency gap becomes positive.

However, the result is still not uniformly solved. Counting and object recognition now show positive conservative gaps, but spatial localization and reasoning still have negative conservative gaps. The model benefits from the spatial prompt and compact frame selection, but its most safety-critical spatial reasoning remains unstable on individual cases.

## Project Implication

This is currently the strongest evidence that DriveMind-VL has a real engineering contribution beyond directly running Qwen2.5-VL:

1. The project established strict visual controls.
2. The initial 5-frame baseline exposed unstable grounding.
3. A targeted spatial prompt plus first/middle/last frame strategy improved both overall performance and conservative visual dependency.
4. The remaining failure modes are now specific enough to guide data curation and SFT.

Next, the best strategy should be used to mine high-value examples:

- positive grounding cases for object recognition and counting;
- hard negative cases where wrong-frame still beats normal;
- spatial localization failures for manual annotation and focused SFT.
