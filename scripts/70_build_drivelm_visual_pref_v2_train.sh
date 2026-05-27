#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA="${DATA:-data/processed/drivelm_train_scene.jsonl}"
WRONG_DATA="${WRONG_DATA:-${DATA%.jsonl}_wrong_image.jsonl}"
BLANK_DATA="${BLANK_DATA:-${DATA%.jsonl}_blank_image.jsonl}"
PREFIX="${PREFIX:-drivelm_train_scene_sft_1200}"
OUTPUT="${OUTPUT:-data/processed/drivelm_visual_pref_v2_train.jsonl}"
SUMMARY="${SUMMARY:-outputs/eval_results/drivelm_visual_pref_v2_train_summary.json}"
MARKDOWN="${MARKDOWN:-docs/drivelm_visual_pref_v2_train_audit.md}"

python src/data/build_drivelm_visual_preference_v1.py \
  --normal_data "${DATA}" \
  --wrong_data "${WRONG_DATA}" \
  --blank_data "${BLANK_DATA}" \
  --cases "outputs/cases/${PREFIX}_visual_control_cases.jsonl" \
  --normal_predictions "outputs/eval_results/${PREFIX}_normal_predictions.jsonl" \
  --text_only_predictions "outputs/eval_results/${PREFIX}_text_only_predictions.jsonl" \
  --wrong_image_predictions "outputs/eval_results/${PREFIX}_wrong_predictions.jsonl" \
  --blank_image_predictions "outputs/eval_results/${PREFIX}_blank_predictions.jsonl" \
  --output "${OUTPUT}" \
  --summary "${SUMMARY}" \
  --markdown "${MARKDOWN}" \
  --seed "${SEED:-20260521}" \
  --max_normal_anchor_pairs "${MAX_NORMAL_ANCHOR_PAIRS:-350}" \
  --max_failure_cases "${MAX_FAILURE_CASES:-1000}" \
  --normal_repair_max_f1 "${NORMAL_REPAIR_MAX_F1:-0.40}" \
  --normal_anchor_weight "${NORMAL_ANCHOR_WEIGHT:-0.35}" \
  --normal_anchor_sft_weight "${NORMAL_ANCHOR_SFT_WEIGHT:-0.8}" \
  --normal_repair_weight "${NORMAL_REPAIR_WEIGHT:-0.8}" \
  --normal_repair_sft_weight "${NORMAL_REPAIR_SFT_WEIGHT:-0.8}" \
  --text_only_weight "${TEXT_ONLY_WEIGHT:-0.60}" \
  --wrong_image_weight "${WRONG_IMAGE_WEIGHT:-0.45}" \
  --blank_image_weight "${BLANK_IMAGE_WEIGHT:-0.75}"
