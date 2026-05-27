#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"$PYTHON_BIN" src/eval/compare_visual_control_runs.py \
  --baseline base_clean_dev \
  --incumbent sft_v2_visual_scale \
  --allow_missing \
  --run base_clean_dev:outputs/eval_results/lingoqa_clean_v2_dev_base_qwen25vl_3b_visual_control_summary.json:outputs/eval_results/lingoqa_clean_v2_dev_base_qwen25vl_3b_visual_control_case_summary.json \
  --run sft_v2_visual_scale:outputs/eval_results/lingoqa_clean_v2_dev_sft_v2_visual_scale_visual_control_summary.json:outputs/eval_results/lingoqa_clean_v2_dev_sft_v2_visual_scale_visual_control_case_summary.json \
  --run pref_v4_guarded_simpo:outputs/eval_results/lingoqa_clean_v2_dev_pref_v4_guarded_simpo_visual_control_summary.json:outputs/eval_results/lingoqa_clean_v2_dev_pref_v4_guarded_simpo_visual_control_case_summary.json \
  --run pref_v5_sweep_best:outputs/eval_results/lingoqa_clean_v2_dev_pref_v5_sweep_best_visual_control_summary.json:outputs/eval_results/lingoqa_clean_v2_dev_pref_v5_sweep_best_visual_control_case_summary.json \
  --output_json outputs/eval_results/lingoqa_visual_control_run_comparison.json \
  --output_csv outputs/eval_results/lingoqa_visual_control_run_comparison.csv \
  --output_md docs/lingoqa_visual_control_run_comparison.md \
  --title "LingoQA Clean Dev Visual-Control Run Comparison" \
  --min_normal_f1 "${MIN_NORMAL_F1:-0.35}" \
  --min_per_case_gap "${MIN_PER_CASE_GAP:--0.01}"
