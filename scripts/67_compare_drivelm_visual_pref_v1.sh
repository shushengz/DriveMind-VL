#!/usr/bin/env bash
set -euo pipefail

python src/eval/compare_visual_control_runs.py \
  --runs \
  "base=outputs/eval_results/${BASE_PREFIX:-drivelm_dev_scene_base}_visual_control_summary.json" \
  "sft_1200=outputs/eval_results/${SFT_PREFIX:-drivelm_dev_scene_sft_1200}_visual_control_summary.json" \
  "visual_pref_v1=outputs/eval_results/${PREF_PREFIX:-drivelm_dev_scene_visual_pref_v1}_visual_control_summary.json" \
  --baseline base \
  --incumbent sft_1200 \
  --output_json "${OUTPUT_JSON:-outputs/eval_results/drivelm_visual_pref_v1_model_comparison.json}" \
  --output_md "${OUTPUT_MD:-docs/drivelm_visual_pref_v1_model_comparison.md}" \
  --title "DriveLM Visual-Preference V1 Comparison" \
  --min_normal_f1 "${MIN_NORMAL_F1:-0.35}" \
  --min_per_case_gap "${MIN_PER_CASE_GAP:--0.10}"
