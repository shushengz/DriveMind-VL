# LingoQA 100-Sample Strict Visual Control Report

## Setup

- Dataset: 100 LingoQA evaluation questions sampled from the official 500-question set.
- Model: Qwen2.5-VL-3B-Instruct.
- Training: none.
- Inputs compared:
  - text-only;
  - normal 5-frame;
  - wrong 5-frame from another sample;
  - blank 5-frame.

## Overall Results

| Setting | JSON Validity | Answer F1 | Lingo-Judge | Pass Rate | Avg Reward |
|---|---:|---:|---:|---:|---:|
| text-only | 1.0000 | 0.1560 | 0.3200 | 0.2600 | 0.4282 |
| normal 5-frame | 1.0000 | 0.2364 | 0.5400 | 0.3700 | 0.4483 |
| wrong 5-frame | 1.0000 | 0.1830 | 0.4400 | 0.3100 | 0.4315 |
| blank 5-frame | 1.0000 | 0.1451 | 0.3400 | 0.2100 | 0.4246 |

Setting-level visual dependency:

- normal vs text-only: `+0.0804` F1
- normal vs wrong-frame: `+0.0534` F1
- normal vs blank-frame: `+0.0913` F1

Lingo-Judge visual dependency:

- normal vs text-only: `+0.2200`
- normal vs wrong-frame: `+0.1000`
- normal vs blank-frame: `+0.2000`

## Per-Case Conservative Gap

The stricter per-case metric compares each normal prediction against the best of its three controls:

```text
gap = normal_f1 - max(text_only_f1, wrong_frame_f1, blank_frame_f1)
```

Overall:

```json
{
  "count": 100,
  "normal_f1": 0.2364,
  "text_only_f1": 0.1560,
  "wrong_image_f1": 0.1830,
  "blank_image_f1": 0.1451,
  "control_max_f1": 0.2695,
  "visual_dependency_gap": -0.0332,
  "positive_gap_rate": 0.1700
}
```

By capability:

| Capability | Count | Normal F1 | Control Max F1 | Conservative Gap |
|---|---:|---:|---:|---:|
| counting | 21 | 0.1336 | 0.0836 | +0.0500 |
| object_recognition | 20 | 0.3260 | 0.2930 | +0.0330 |
| other | 20 | 0.3531 | 0.4425 | -0.0894 |
| reasoning_world_knowledge | 17 | 0.1290 | 0.2684 | -0.1394 |
| spatial_localization | 20 | 0.2476 | 0.2911 | -0.0435 |
| weather_road_condition | 2 | 0.0513 | 0.0513 | 0.0000 |

## Interpretation

At the setting level, the correct 5-frame input beats text-only, wrong-frame, and blank-frame controls. This is evidence that visual information helps.

However, the conservative per-case gap is negative. This means the model's improvement is not stable on a sample-by-sample basis: for many examples, a control condition accidentally matches the reference as well as or better than the true visual input. The model is therefore not reliably visually grounded yet.

Lingo-Judge gives a stronger positive setting-level signal than token-F1. This suggests token-F1 underestimates semantically correct paraphrases, but it does not remove the per-case instability observed under the stricter control-max analysis.

The strongest positive control evidence is in counting and object recognition. Spatial localization is still weak and unstable, even though it is central to driving VQA.

## Project Implication

This is a more rigorous result than the 500-question text/single/5-frame comparison alone. It supports a nuanced conclusion:

DriveMind-VL with Qwen2.5-VL-3B has real visual signal on LingoQA, but current prompting and frame packing do not produce robust per-case visual dependency.

## Next Step

Use the Lingo-Judge cases and token-F1 bad cases together to build a manual error taxonomy.
