# External Benchmark Audit

Input: `data/processed/drivelm_dev_scene.jsonl`
Samples: `600`

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
| counting | 11 |
| object_recognition | 100 |
| reasoning_world_knowledge | 57 |
| scene_completeness | 34 |
| spatial_localization | 398 |

## By Source

| source | count |
|---|---:|
| drivelm | 600 |

## Top Categories

| category | count |
|---|---:|
| perception | 259 |
| prediction | 189 |
| planning | 142 |
| behavior | 10 |
