#!/usr/bin/env bash
set -euo pipefail

for name in normal text_only wrong_image blank_image; do
  python src/eval/eval_external_vqa_breakdown.py \
    --predictions "outputs/eval_results/intelli_${name}_predictions.jsonl" \
    --output "outputs/eval_results/intelli_${name}_external_vqa_breakdown.json"
done

python src/eval/compare_visual_ablation_by_case.py \
  --normal outputs/eval_results/intelli_normal_predictions.jsonl \
  --text_only outputs/eval_results/intelli_text_only_predictions.jsonl \
  --wrong_image outputs/eval_results/intelli_wrong_image_predictions.jsonl \
  --blank_image outputs/eval_results/intelli_blank_image_predictions.jsonl \
  --output outputs/eval_results/intelli_visual_ablation_case_summary.json \
  --cases_output outputs/cases/intelli_visual_ablation_cases.jsonl
