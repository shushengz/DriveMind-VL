# DriveLM Visual Preference V1 Audit

Output: `data/processed/drivelm_visual_pref_v2_train.jsonl`
Pairs: `2957`

## Pair Types

| name | count |
|---|---:|
| control_refusal_over_prediction | 1957 |
| normal_repair_gold_over_bad_prediction | 650 |
| normal_anchor_gold_over_refusal | 350 |

## Train Input Modes

| name | count |
|---|---:|
| normal | 1000 |
| blank_image | 653 |
| text_only | 652 |
| wrong_image | 652 |

## Capabilities

| name | count |
|---|---:|
| spatial_localization | 1842 |
| object_recognition | 550 |
| reasoning_world_knowledge | 354 |
| scene_completeness | 170 |
| counting | 37 |
| weather_road_condition | 4 |

## Selected Failure Cases

| name | count |
|---|---:|
| wrong_image_invariant_low | 297 |
| text_beats_normal | 204 |
| wrong_image_beats_normal | 95 |
| blank_beats_normal | 57 |
| normal_low | 24 |

## Counters

| name | count |
|---|---:|
| blank_image_control_added | 653 |
| text_only_control_added | 652 |
| wrong_image_control_added | 652 |
| normal_repair_added | 650 |
| normal_anchor_added | 350 |
| text_only_control_skipped_already_refusal | 1 |
| wrong_image_control_skipped_already_refusal | 1 |

## Notes

- Normal-anchor pairs preserve the DriveLM answer style and reduce refusal drift.
- Normal-repair pairs teach the model to prefer gold answers over its own bad normal-image predictions.
- Control-refusal pairs teach text-only, wrong-image, and blank-image prompts to refuse scene-specific answers.
- If this file was built from dev predictions, use it as a diagnostic smoke artifact only; final DPO data should be rebuilt from train-scene predictions to avoid evaluation leakage.
