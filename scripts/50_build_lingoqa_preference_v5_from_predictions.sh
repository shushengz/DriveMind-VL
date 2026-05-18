#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PREFIX="${PREFIX:-lingoqa_clean_v2_train_sft_v2_visual_scale_for_pref}"

python src/data/build_lingoqa_preference_v5_from_predictions.py \
  --normal "${NORMAL_DATA:-data/processed/lingoqa_clean_v2_train.jsonl}" \
  --wrong "${WRONG_DATA:-data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl}" \
  --blank "${BLANK_DATA:-data/processed/lingoqa_clean_v2_train_blank_frame.jsonl}" \
  --normal_predictions "${NORMAL_PREDICTIONS:-outputs/eval_results/${PREFIX}_normal_predictions.jsonl}" \
  --text_predictions "${TEXT_PREDICTIONS:-outputs/eval_results/${PREFIX}_text_only_predictions.jsonl}" \
  --wrong_predictions "${WRONG_PREDICTIONS:-outputs/eval_results/${PREFIX}_wrong_predictions.jsonl}" \
  --blank_predictions "${BLANK_PREDICTIONS:-outputs/eval_results/${PREFIX}_blank_predictions.jsonl}" \
  --output "${OUTPUT:-data/processed/lingoqa_preference_v5_from_sft_v2_train_predictions.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/lingoqa_preference_v5_from_sft_v2_train_predictions_summary.json}" \
  --seed "${SEED:-20260517}" \
  --normal_weight "${NORMAL_WEIGHT:-1.0}" \
  --text_only_weight "${TEXT_ONLY_WEIGHT:-0.30}" \
  --blank_image_weight "${BLANK_IMAGE_WEIGHT:-0.40}" \
  --wrong_image_weight "${WRONG_IMAGE_WEIGHT:-0.60}" \
  --normal_sft_anchor_weight "${NORMAL_SFT_ANCHOR_WEIGHT:-1.0}" \
  --normal_bad_f1_threshold "${NORMAL_BAD_F1_THRESHOLD:-0.15}" \
  --min_control_f1 "${MIN_CONTROL_F1:-0.05}"
