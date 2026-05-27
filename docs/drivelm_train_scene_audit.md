# External Benchmark Audit

Input: `data/processed/drivelm_train_scene.jsonl`
Samples: `2400`

## Checks

| check | value |
|---|---:|
| missing id | 0 |
| duplicate id | 0 |
| missing instruction | 0 |
| missing answer | 0 |
| non external_vqa task | 0 |
| missing media | 0 |
| media paths not found | 0 |
| avg image paths | 6.0 |

## By Capability

| capability | count |
|---|---:|
| counting | 29 |
| object_recognition | 372 |
| reasoning_world_knowledge | 238 |
| scene_completeness | 185 |
| spatial_localization | 1575 |
| weather_road_condition | 1 |

## By Source

| source | count |
|---|---:|
| drivelm | 2400 |

## Top Categories

| category | count |
|---|---:|
| perception | 1065 |
| prediction | 769 |
| planning | 543 |
| behavior | 23 |
