#!/usr/bin/env bash
set -euo pipefail

BASE_PREFIX="${BASE_PREFIX:-drivelm_dev_scene_base}"
ADAPTER_PREFIX="${ADAPTER_PREFIX:-drivelm_dev_scene_sft}"
OUTPUT_JSON="${OUTPUT_JSON:-outputs/eval_results/drivelm_dev_scene_model_comparison.json}"
OUTPUT_MD="${OUTPUT_MD:-docs/drivelm_dev_scene_model_comparison.md}"

python src/eval/compare_visual_control_runs.py \
  --runs \
  "base=outputs/eval_results/${BASE_PREFIX}_visual_control_summary.json" \
  "drivelm_sft=outputs/eval_results/${ADAPTER_PREFIX}_visual_control_summary.json" \
  --baseline base \
  --output_json "${OUTPUT_JSON}" \
  --output_md "${OUTPUT_MD}" \
  --title "DriveLM Scene-Heldout Visual-Control Comparison" \
  --min_normal_f1 "${MIN_NORMAL_F1:-0.25}" \
  --min_per_case_gap "${MIN_PER_CASE_GAP:--0.05}"
