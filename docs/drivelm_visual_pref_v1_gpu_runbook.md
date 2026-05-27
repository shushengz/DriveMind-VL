# DriveLM Visual Preference V1 GPU Runbook

Run this after switching the server to GPU mode. The no-GPU diagnostic builder has already validated the preference format; the final training data must be rebuilt from train-scene predictions to avoid dev leakage.

## 1. Generate Train-Scene Visual-Control Predictions

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

MAX_SAMPLES=1200 \
bash scripts/68_eval_drivelm_train_scene_sft_for_pref.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
2>&1 | tee outputs/logs/drivelm_train_scene_sft_1200_for_pref.log
```

This produces:

- `outputs/eval_results/drivelm_train_scene_sft_1200_normal_predictions.jsonl`
- `outputs/eval_results/drivelm_train_scene_sft_1200_text_only_predictions.jsonl`
- `outputs/eval_results/drivelm_train_scene_sft_1200_wrong_predictions.jsonl`
- `outputs/eval_results/drivelm_train_scene_sft_1200_blank_predictions.jsonl`
- `outputs/cases/drivelm_train_scene_sft_1200_visual_control_cases.jsonl`

## 2. Build Final Train Preference Data

This step does not need GPU, but run it after step 1 while the files are present.

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

bash scripts/69_build_drivelm_visual_pref_v1_train.sh
```

Expected output:

- `data/processed/drivelm_visual_pref_v1_train.jsonl`
- `docs/drivelm_visual_pref_v1_train_audit.md`

## 3. Train Visual-Preference Adapter

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

TRAIN_FILE=data/processed/drivelm_visual_pref_v1_train.jsonl \
MAX_PAIRS=800 \
MAX_STEPS=24 \
OUTPUT_DIR=outputs/checkpoints/qwen25vl_3b_drivelm_visual_pref_v1 \
bash scripts/65_train_drivelm_visual_pref_v1.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
2>&1 | tee outputs/logs/train_drivelm_visual_pref_v1.log
```

## 4. Evaluate And Compare

```bash
cd /root/autodl-tmp/DriveMind-VL
source /root/miniconda3/bin/activate drivemind-vl

PREFIX=drivelm_dev_scene_visual_pref_v1 \
ADAPTER=outputs/checkpoints/qwen25vl_3b_drivelm_visual_pref_v1/checkpoint-step-000024 \
MAX_SAMPLES=300 \
bash scripts/66_eval_drivelm_visual_pref_v1.sh /root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct \
2>&1 | tee outputs/logs/drivelm_dev_scene_visual_pref_v1.log

bash scripts/67_compare_drivelm_visual_pref_v1.sh
```

## Acceptance Criteria

- normal F1 stays at or above `0.35`;
- blank-image F1 drops below normal F1;
- text-only F1 drops below normal F1;
- case-level visual dependency gap improves from `-0.1576` to better than `-0.10`.
