# Visual Grounding Ablation Report

This report records the first image-dependence check for DriveMind-VL on the 7-sample IntelliCockpitBench smoke set.

## Goal

The previous external smoke test showed that Qwen2.5-VL-3B can follow DriveMind JSON schema on public cockpit/driving images, but its fine-grained VQA agreement is weak. This ablation checks whether the model is actually using the image.

## Settings

All settings use the same 7 IntelliCockpitBench samples and Qwen2.5-VL-3B base model.

- `normal`: original image.
- `text_only`: no image passed to the model.
- `wrong_image`: image paths cyclically shifted across samples.
- `blank_image`: all samples use a generated blank gray image.

## Commands

```bash
bash scripts/18_run_intelli_visual_ablation.sh \
  /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
  third_party/IntelliCockpitBench
```

## Results

| Setting | JSON Validity | External Answer F1 | Avg Reward | Failed Cases |
|---|---:|---:|---:|---:|
| normal | 1.0000 | 0.2070 | 0.4371 | 4 / 7 |
| text_only | 1.0000 | 0.0801 | 0.4133 | 6 / 7 |
| wrong_image | 1.0000 | 0.1135 | 0.4162 | 5 / 7 |
| blank_image | 1.0000 | 0.1766 | 0.4423 | 5 / 7 |

## Interpretation

The normal-image run performs better than text-only and wrong-image runs, so the model is using some visual signal. However, the gap is small on this 7-sample set, and blank-image performance remains close to the normal run. This means the current small sample is not enough to make a strong visual-grounding claim.

The result supports a stricter next step: expand IntelliCockpitBench samples and break external VQA into subskills such as counting, object recognition, spatial localization, weather/road condition, and scene completeness. Text-only and wrong-image controls should remain mandatory for future external reports.

## Updated Metric Note

`external_vqa` reward was updated so its task component uses answer token-F1 instead of only task-name matching. This makes `avg_reward` less misleading for external VQA, but `external_answer_f1` remains the primary metric for this smoke test.
