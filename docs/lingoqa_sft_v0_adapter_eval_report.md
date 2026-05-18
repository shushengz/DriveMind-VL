# LingoQA SFT-v0 Adapter Strict Visual-Control Eval

## Setup

- Base model: Qwen2.5-VL-3B-Instruct.
- Adapter: `outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v0_smoke`.
- Training data: 25 reviewed `keep` samples from `DriveMind-LingoQA-SFT-v0`.
- Evaluation data: 100 LingoQA control samples.
- Strategy: `prompt_variant=spatial`, `frame_strategy=first_middle_last`, `max_images=3`.
- Controls: text-only, normal 3-frame, wrong 3-frame, blank 3-frame.

## Adapter Results

| Setting | JSON Validity | Answer F1 | Pass Rate | Avg Reward |
|---|---:|---:|---:|---:|
| text-only | 1.0000 | 0.1634 | 0.3300 | 0.4327 |
| normal 3-frame | 0.9900 | 0.2470 | 0.4100 | 0.4462 |
| wrong 3-frame | 1.0000 | 0.2106 | 0.3300 | 0.4390 |
| blank 3-frame | 1.0000 | 0.1773 | 0.2800 | 0.4327 |

Setting-level visual dependency:

```text
normal_f1 - max(text_only_f1, wrong_image_f1, blank_image_f1)
= 0.2470 - 0.2106
= +0.0364
```

Per-case conservative gap:

```json
{
  "normal_f1": 0.2470,
  "text_only_f1": 0.1634,
  "wrong_image_f1": 0.2106,
  "blank_image_f1": 0.1773,
  "control_max_f1": 0.2882,
  "visual_dependency_gap": -0.0412,
  "positive_gap_rate": 0.1500
}
```

## Pre/Post Comparison

| Metric | Base Best Strategy | SFT-v0 Adapter | Change |
|---|---:|---:|---:|
| normal F1 | 0.2912 | 0.2470 | -0.0442 |
| text-only F1 | 0.1448 | 0.1634 | +0.0186 |
| wrong-image F1 | 0.2095 | 0.2106 | +0.0011 |
| blank-image F1 | 0.1349 | 0.1773 | +0.0424 |
| setting-level gap | +0.0817 | +0.0364 | -0.0453 |
| per-case conservative gap | +0.0212 | -0.0412 | -0.0624 |
| positive-gap rate | 0.3100 | 0.1500 | -0.1600 |

## Capability Breakdown

| Capability | Base Normal F1 | Adapter Normal F1 | Base Gap | Adapter Gap |
|---|---:|---:|---:|---:|
| counting | 0.1738 | 0.2881 | +0.0902 | +0.0451 |
| object_recognition | 0.3791 | 0.2959 | +0.0927 | -0.0508 |
| other | 0.4031 | 0.3580 | -0.0149 | -0.0413 |
| reasoning_world_knowledge | 0.2034 | 0.1156 | -0.0553 | -0.0682 |
| spatial_localization | 0.3134 | 0.1468 | -0.0195 | -0.0798 |
| weather_road_condition | 0.0513 | 0.3333 | 0.0000 | -0.2349 |

## Interpretation

The adapter is technically valid: it loads, produces mostly valid JSON, and completes all four visual-control runs. However, it does not improve the strict grounding objective.

The most important warning signs are:

1. Normal 3-frame F1 drops from `0.2912` to `0.2470`.
2. Text-only and blank-image scores increase, which means the adapter may have strengthened language-prior behavior.
3. Per-case conservative gap drops from `+0.0212` to `-0.0412`.
4. Spatial localization drops sharply from `0.3134` to `0.1468`.

This suggests the 25-sample SFT-v0 is too small and too skewed toward anti-hallucination / wrong-image-confound samples. It is useful as a pipeline smoke test, but not as a model-quality improvement.

## Decision

Do not treat this adapter as the current best model. Keep the base Qwen2.5-VL-3B with `spatial + first/middle/last 3-frame` as the current best inference strategy.

Next data step:

1. Do not increase epochs on the same 25 samples.
2. Rewrite the 7 `fix` samples.
3. Add more positive visual-grounding and spatial-localization samples.
4. Reduce the proportion of `wrong_image_confound` in the SFT mix, or train those with an explicit contrastive/rejection objective rather than plain SFT.
5. Rebuild a 50-80 sample SFT-v1 and rerun the same strict visual-control evaluation.
