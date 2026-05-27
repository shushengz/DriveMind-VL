#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATA="${DATA:-data/processed/drivelm_dev_scene.jsonl}"
WRONG_DATA="${WRONG_DATA:-${DATA%.jsonl}_wrong_image.jsonl}"
BLANK_DATA="${BLANK_DATA:-${DATA%.jsonl}_blank_image.jsonl}"
PREFIX="${PREFIX:-drivelm_dev_scene_sft_1200}"
OUTPUT="${OUTPUT:-data/processed/drivelm_visual_pref_v1_dev_diagnostic.jsonl}"
SUMMARY="${SUMMARY:-outputs/eval_results/drivelm_visual_pref_v1_dev_diagnostic_summary.json}"
MARKDOWN="${MARKDOWN:-docs/drivelm_visual_pref_v1_dev_diagnostic.md}"

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
  --max_normal_anchor_pairs "${MAX_NORMAL_ANCHOR_PAIRS:-300}" \
  --max_failure_cases "${MAX_FAILURE_CASES:-240}"
