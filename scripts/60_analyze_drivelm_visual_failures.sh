#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

PREFIX="${PREFIX:-drivelm_eval_v7_step24}"
TITLE="${TITLE:-DriveLM V7 Visual-Control Failure Analysis}"

python src/eval/analyze_visual_control_failures.py \
  --cases "outputs/cases/${PREFIX}_visual_control_cases.jsonl" \
  --normal_predictions "outputs/eval_results/${PREFIX}_normal_predictions.jsonl" \
  --text_only_predictions "outputs/eval_results/${PREFIX}_text_only_predictions.jsonl" \
  --wrong_image_predictions "outputs/eval_results/${PREFIX}_wrong_predictions.jsonl" \
  --blank_image_predictions "outputs/eval_results/${PREFIX}_blank_predictions.jsonl" \
  --output_json "outputs/eval_results/${PREFIX}_failure_analysis.json" \
  --output_csv "outputs/eval_results/${PREFIX}_failure_cases.csv" \
  --output_md "docs/${PREFIX}_failure_analysis.md" \
  --title "${TITLE}"
