# DriveLM Scene-Level Split Report

Input: `data/external/drivelm/v1_1_train_nus.json`
Train: `data/processed/drivelm_train_scene.jsonl`
Dev: `data/processed/drivelm_dev_scene.jsonl`
Seed: `42`

## Leakage Check

Scene overlap count: `0`

## Counts

| split | rows | scenes | avg rows / scene |
|---|---:|---:|---:|
| all | 377956 | 696 | 543.0402 |
| train | 2400 | 639 | 3.7559 |
| dev | 600 | 30 | 20.0 |

## Train By Capability

| name | count |
|---|---:|
| spatial_localization | 1575 |
| object_recognition | 372 |
| reasoning_world_knowledge | 238 |
| scene_completeness | 185 |
| counting | 29 |
| weather_road_condition | 1 |

## Train By Category

| name | count |
|---|---:|
| perception | 1065 |
| prediction | 769 |
| planning | 543 |
| behavior | 23 |

## Dev By Capability

| name | count |
|---|---:|
| spatial_localization | 398 |
| object_recognition | 100 |
| reasoning_world_knowledge | 57 |
| scene_completeness | 34 |
| counting | 11 |

## Dev By Category

| name | count |
|---|---:|
| perception | 259 |
| prediction | 189 |
| planning | 142 |
| behavior | 10 |
