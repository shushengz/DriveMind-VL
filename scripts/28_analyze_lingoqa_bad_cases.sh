#!/usr/bin/env bash
set -euo pipefail

python src/eval/analyze_lingoqa_bad_cases.py \
  --normal outputs/eval_results/lingoqa_qwen25vl_3b_100_control_normal_5frame_predictions.jsonl \
  --text_only outputs/eval_results/lingoqa_qwen25vl_3b_100_control_text_only_predictions.jsonl \
  --wrong_image outputs/eval_results/lingoqa_qwen25vl_3b_100_control_wrong_5frame_predictions.jsonl \
  --blank_image outputs/eval_results/lingoqa_qwen25vl_3b_100_control_blank_5frame_predictions.jsonl \
  --dataset data/processed/drivemind_lingoqa_eval_100_control.jsonl \
  --csv_output outputs/cases/lingoqa_bad_case_taxonomy.csv \
  --json_output outputs/eval_results/lingoqa_bad_case_taxonomy_summary.json \
  --report_output docs/lingoqa_bad_case_analysis.md
