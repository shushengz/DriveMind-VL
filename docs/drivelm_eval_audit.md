# External Benchmark Audit

Input: `data/processed/drivelm_eval.jsonl`
Samples: `300`

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
| counting | 7 |
| object_recognition | 59 |
| reasoning_world_knowledge | 52 |
| scene_completeness | 29 |
| spatial_localization | 152 |
| weather_road_condition | 1 |

## By Source

| source | count |
|---|---:|
| drivelm | 300 |

## Top Categories

| category | count |
|---|---:|
| perception | 143 |
| prediction | 99 |
| planning | 55 |
| behavior | 3 |
