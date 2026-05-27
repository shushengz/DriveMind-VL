# Public Benchmark Runbook

This runbook prepares public benchmark data without using GPU. GPU is only needed after the audit passes.

## DriveLM

Official source: https://github.com/OpenDriveLab/DriveLM

DriveLM-nuScenes stores QA as a nested JSON:

```text
scene_token -> key_frames -> frame_token -> QA -> perception/prediction/planning/behavior
```

The converter flattens each QA item into one DriveMind `external_vqa` sample and preserves:

- `scene_token`
- `sample_token`
- `category`, such as perception, prediction, planning, behavior
- `image_paths`
- `scene_description`
- `key_object_infos`

Expected local placement after download:

```text
data/external/drivelm/
  v1_0_train_nus.json
  nuscenes/
    samples/
```

Prepare a small no-GPU subset:

```bash
SOURCE=drivelm \
INPUT=data/external/drivelm/v1_0_train_nus.json \
IMAGE_ROOT=data/external/drivelm \
SPLIT=eval \
LIMIT=300 \
OUTPUT=data/processed/drivelm_eval.jsonl \
bash scripts/58_prepare_public_benchmark_eval.sh
```

## NuScenes-QA

Official source: https://github.com/qiantianwen/NuScenes-QA

The official README expects downloaded question files under:

```text
data/questions/
  NuScenes_train_questions.json
  NuScenes_val_questions.json
```

Prepare a small no-GPU subset:

```bash
SOURCE=nuscenes_qa \
INPUT=data/external/nuscenes_qa/questions/NuScenes_val_questions.json \
IMAGE_ROOT=data/external/nuscenes_qa \
SPLIT=eval \
LIMIT=300 \
OUTPUT=data/processed/nuscenes_qa_eval.jsonl \
bash scripts/58_prepare_public_benchmark_eval.sh
```

If using original nuScenes images, set `IMAGE_ROOT` to the directory that makes the image paths in the question file resolvable.

## Audit Acceptance

Before GPU inference, check the generated audit markdown under `docs/`.

Acceptable for a first small run:

- `missing instruction = 0`
- `missing answer = 0`
- `non external_vqa task = 0`
- `duplicate id = 0`
- `missing media = 0`
- `media paths not found = 0`, unless the dataset uses remote URLs or features instead of images

The capability table should not collapse entirely into `other`. If it does, improve taxonomy before spending GPU.

## GPU Evaluation

Base model:

```bash
DATA=data/processed/drivelm_eval.jsonl \
PREFIX=drivelm_eval_qwen25vl_3b_base \
MAX_SAMPLES=300 \
bash scripts/55_eval_external_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

SFT-v2:

```bash
DATA=data/processed/drivelm_eval.jsonl \
PREFIX=drivelm_eval_sft_v2 \
ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale \
MAX_SAMPLES=300 \
bash scripts/55_eval_external_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

Current best preference adapter:

```bash
DATA=data/processed/drivelm_eval.jsonl \
PREFIX=drivelm_eval_v5c_step20 \
ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5c_step90_stronger_controls/checkpoint-step-000020 \
MAX_SAMPLES=300 \
bash scripts/55_eval_external_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

Future v7:

```bash
DATA=data/processed/drivelm_eval.jsonl \
PREFIX=drivelm_eval_v7_step12 \
ADAPTER=outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v7_grounded/checkpoint-step-000012 \
MAX_SAMPLES=300 \
bash scripts/55_eval_external_visual_controls.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct
```

## Benchmark Table Template

| dataset | model | normal F1 | text-only F1 | wrong-image F1 | blank-image F1 | setting gap | per-case gap | spatial normal F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LingoQA dev | Qwen2.5-VL-3B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| LingoQA dev | SFT-v2 | 0.3680 | 0.2873 | 0.2965 | 0.2564 | 0.0715 | -0.0070 | TBD |
| LingoQA dev | v5c step20 | 0.3827 | 0.2914 | 0.2982 | 0.2826 | 0.0845 | 0.0051 | TBD |
| LingoQA dev | v7 step12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DriveLM subset | Qwen2.5-VL-3B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DriveLM subset | SFT-v2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DriveLM subset | v7 step12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NuScenes-QA subset | Qwen2.5-VL-3B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NuScenes-QA subset | SFT-v2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NuScenes-QA subset | v7 step12 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
