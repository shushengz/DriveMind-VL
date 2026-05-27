# DriveLM Visual Preference V1 Audit

Output: `data/processed/drivelm_visual_pref_v1_dev_diagnostic.jsonl`
Pairs: `1005`

## Pair Types

| name | count |
|---|---:|
| control_refusal_over_prediction | 533 |
| normal_anchor_gold_over_refusal | 300 |
| normal_repair_gold_over_bad_prediction | 172 |

## Train Input Modes

| name | count |
|---|---:|
| normal | 472 |
| text_only | 178 |
| wrong_image | 178 |
| blank_image | 177 |

## Capabilities

| name | count |
|---|---:|
| spatial_localization | 638 |
| object_recognition | 178 |
| reasoning_world_knowledge | 121 |
| scene_completeness | 51 |
| counting | 17 |

## Selected Failure Cases

| name | count |
|---|---:|
| wrong_image_invariant_low | 82 |
| text_beats_normal | 60 |
| wrong_image_beats_normal | 25 |
| blank_beats_normal | 11 |
| normal_low | 5 |

## Counters

| name | count |
|---|---:|
| normal_anchor_added | 300 |
| text_only_control_added | 178 |
| wrong_image_control_added | 178 |
| blank_image_control_added | 177 |
| normal_repair_added | 172 |
| blank_image_control_skipped_already_refusal | 1 |

## Notes

- Normal-anchor pairs preserve the DriveLM answer style and reduce refusal drift.
- Normal-repair pairs teach the model to prefer gold answers over its own bad normal-image predictions.
- Control-refusal pairs teach text-only, wrong-image, and blank-image prompts to refuse scene-specific answers.
- If this file was built from dev predictions, use it as a diagnostic smoke artifact only; final DPO data should be rebuilt from train-scene predictions to avoid evaluation leakage.
