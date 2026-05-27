#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PREFIX="${PREFIX:-lingoqa_clean_v2_train_sft_v2_visual_scale_for_pref}"

python src/data/build_lingoqa_preference_v7_grounded.py \
  --normal "${NORMAL_DATA:-data/processed/lingoqa_clean_v2_train.jsonl}" \
  --wrong "${WRONG_DATA:-data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl}" \
  --blank "${BLANK_DATA:-data/processed/lingoqa_clean_v2_train_blank_frame.jsonl}" \
  --normal_predictions "${NORMAL_PREDICTIONS:-outputs/eval_results/${PREFIX}_normal_predictions.jsonl}" \
  --text_predictions "${TEXT_PREDICTIONS:-outputs/eval_results/${PREFIX}_text_only_predictions.jsonl}" \
  --wrong_predictions "${WRONG_PREDICTIONS:-outputs/eval_results/${PREFIX}_wrong_predictions.jsonl}" \
  --blank_predictions "${BLANK_PREDICTIONS:-outputs/eval_results/${PREFIX}_blank_predictions.jsonl}" \
  --output "${OUTPUT:-data/processed/lingoqa_preference_v7_grounded.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/lingoqa_preference_v7_grounded_summary.json}" \
  --seed "${SEED:-20260519}" \
  --normal_weight "${NORMAL_WEIGHT:-1.0}" \
  --text_only_weight "${TEXT_ONLY_WEIGHT:-0.20}" \
  --blank_image_weight "${BLANK_IMAGE_WEIGHT:-0.25}" \
  --wrong_image_weight "${WRONG_IMAGE_WEIGHT:-0.35}" \
  --normal_sft_anchor_weight "${NORMAL_SFT_ANCHOR_WEIGHT:-1.5}" \
  --normal_bad_f1_threshold "${NORMAL_BAD_F1_THRESHOLD:-0.15}" \
  --min_normal_prediction_f1_for_control "${MIN_NORMAL_PREDICTION_F1_FOR_CONTROL:-0.45}" \
  --max_control_prediction_f1 "${MAX_CONTROL_PREDICTION_F1:-0.35}" \
  --min_visual_contrast_gap "${MIN_VISUAL_CONTRAST_GAP:-0.20}" \
  --control_modes "${CONTROL_MODES:-wrong_image,blank_image}" \
  --max_text_only_pairs "${MAX_TEXT_ONLY_PAIRS:-0}" \
  --max_blank_image_pairs "${MAX_BLANK_IMAGE_PAIRS:-80}" \
  --max_wrong_image_pairs "${MAX_WRONG_IMAGE_PAIRS:-80}" \
  --max_control_pairs_per_capability "${MAX_CONTROL_PAIRS_PER_CAPABILITY:-60}" \
  --capability_sft_anchor_weights "${CAPABILITY_SFT_ANCHOR_WEIGHTS:-spatial_localization=2.25,counting=1.75,object_recognition=1.5,reasoning_world_knowledge=1.75}" \
  --capability_control_weight_multipliers "${CAPABILITY_CONTROL_WEIGHT_MULTIPLIERS:-spatial_localization=0.65,counting=0.75,object_recognition=0.85}"
