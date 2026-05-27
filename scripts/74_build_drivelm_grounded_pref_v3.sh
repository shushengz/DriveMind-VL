#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PREFIX="${PREFIX:-drivelm_train_scene_sft_1200}"

python src/data/build_drivelm_grounded_preference_v3.py \
  --normal_data "${NORMAL_DATA:-data/processed/drivelm_train_scene.jsonl}" \
  --wrong_data "${WRONG_DATA:-data/processed/drivelm_train_scene_wrong_image.jsonl}" \
  --blank_data "${BLANK_DATA:-data/processed/drivelm_train_scene_blank_image.jsonl}" \
  --cases "${CASES:-outputs/cases/${PREFIX}_visual_control_cases.jsonl}" \
  --text_only_predictions "${TEXT_ONLY_PREDICTIONS:-outputs/eval_results/${PREFIX}_text_only_predictions.jsonl}" \
  --wrong_image_predictions "${WRONG_IMAGE_PREDICTIONS:-outputs/eval_results/${PREFIX}_wrong_predictions.jsonl}" \
  --blank_image_predictions "${BLANK_IMAGE_PREDICTIONS:-outputs/eval_results/${PREFIX}_blank_predictions.jsonl}" \
  --output "${OUTPUT:-data/processed/drivelm_grounded_pref_v3_train.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/drivelm_grounded_pref_v3_train_summary.json}" \
  --seed "${SEED:-20260521}" \
  --control_modes "${CONTROL_MODES:-wrong_image,blank_image}" \
  --max_normal_anchor_pairs "${MAX_NORMAL_ANCHOR_PAIRS:-600}" \
  --max_control_pairs_per_capability "${MAX_CONTROL_PAIRS_PER_CAPABILITY:-120}" \
  --min_normal_f1 "${MIN_NORMAL_F1:-0.40}" \
  --max_control_f1 "${MAX_CONTROL_F1:-0.30}" \
  --min_visual_contrast_gap "${MIN_VISUAL_CONTRAST_GAP:-0.20}" \
  --normal_weight "${NORMAL_WEIGHT:-1.0}" \
  --text_only_weight "${TEXT_ONLY_WEIGHT:-0.15}" \
  --wrong_image_weight "${WRONG_IMAGE_WEIGHT:-0.30}" \
  --blank_image_weight "${BLANK_IMAGE_WEIGHT:-0.25}" \
  --normal_sft_anchor_weight "${NORMAL_SFT_ANCHOR_WEIGHT:-1.5}" \
  --capability_sft_anchor_weights "${CAPABILITY_SFT_ANCHOR_WEIGHTS:-spatial_localization=2.0,reasoning_world_knowledge=1.8,object_recognition=1.6,counting=1.6}" \
  --capability_control_weight_multipliers "${CAPABILITY_CONTROL_WEIGHT_MULTIPLIERS:-spatial_localization=0.65,reasoning_world_knowledge=0.9,object_recognition=0.85,counting=0.8}"
