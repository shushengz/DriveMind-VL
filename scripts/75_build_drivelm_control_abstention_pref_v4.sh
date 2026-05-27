#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python src/data/build_drivelm_control_abstention_pref_v4.py \
  --normal_data "${NORMAL_DATA:-data/processed/drivelm_train_scene.jsonl}" \
  --wrong_data "${WRONG_DATA:-data/processed/drivelm_train_scene_wrong_image_hard.jsonl}" \
  --blank_data "${BLANK_DATA:-data/processed/drivelm_train_scene_blank_image_v4.jsonl}" \
  --output "${OUTPUT:-data/processed/drivelm_control_abstention_pref_v4_train.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/drivelm_control_abstention_pref_v4_train_summary.json}" \
  --seed "${SEED:-20260522}" \
  --control_modes "${CONTROL_MODES:-text_only,wrong_image,blank_image}" \
  --max_normal_pairs "${MAX_NORMAL_PAIRS:-900}" \
  --max_control_pairs_per_mode "${MAX_CONTROL_PAIRS_PER_MODE:-700}" \
  --max_control_pairs_per_capability "${MAX_CONTROL_PAIRS_PER_CAPABILITY:-220}" \
  --normal_weight "${NORMAL_WEIGHT:-1.0}" \
  --text_only_weight "${TEXT_ONLY_WEIGHT:-0.45}" \
  --wrong_image_weight "${WRONG_IMAGE_WEIGHT:-0.45}" \
  --blank_image_weight "${BLANK_IMAGE_WEIGHT:-0.55}" \
  --normal_sft_anchor_weight "${NORMAL_SFT_ANCHOR_WEIGHT:-1.2}" \
  --control_sft_anchor_weight "${CONTROL_SFT_ANCHOR_WEIGHT:-0.8}" \
  --capability_weight_multipliers "${CAPABILITY_WEIGHT_MULTIPLIERS:-spatial_localization=1.15,reasoning_world_knowledge=1.0,object_recognition=1.0,counting=1.0}"
