#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
INTELLI_REPO=${2:-third_party/IntelliCockpitBench}
MAX_SAMPLES=${MAX_SAMPLES:-20}

BASE_DATA=data/processed/drivemind_intelli_sample.jsonl
WRONG_DATA=data/processed/drivemind_intelli_sample_wrong_image.jsonl
BLANK_DATA=data/processed/drivemind_intelli_sample_blank_image.jsonl

python src/data/prepare_intelli_cockpitbench_sample.py \
  --repo_root "${INTELLI_REPO}" \
  --output "${BASE_DATA}" \
  --limit "${MAX_SAMPLES}"

python src/data/make_visual_ablation.py \
  --input "${BASE_DATA}" \
  --output "${WRONG_DATA}" \
  --mode wrong_image

python src/data/make_visual_ablation.py \
  --input "${BASE_DATA}" \
  --output "${BLANK_DATA}" \
  --mode blank_image

run_eval() {
  local name=$1
  local input=$2
  shift 2
  python src/eval/base_infer_qwen25vl.py \
    --input "${input}" \
    --output "outputs/eval_results/intelli_${name}_predictions.jsonl" \
    --model_name_or_path "${MODEL_PATH}" \
    --max_samples "${MAX_SAMPLES}" \
    --max_new_tokens 160 \
    --bf16 \
    "$@"

  python src/eval/eval_by_source.py \
    --predictions "outputs/eval_results/intelli_${name}_predictions.jsonl" \
    --output "outputs/eval_results/intelli_${name}_metrics_by_source.json" \
    --bad_cases_output "outputs/cases/intelli_${name}_bad_cases.jsonl"

  python src/eval/error_analysis.py \
    --predictions "outputs/eval_results/intelli_${name}_predictions.jsonl" \
    --output "outputs/eval_results/intelli_${name}_error_analysis.json" \
    --cases_output "outputs/cases/intelli_${name}_error_cases.jsonl"
}

run_eval normal "${BASE_DATA}"
run_eval text_only "${BASE_DATA}" --text_only
run_eval wrong_image "${WRONG_DATA}"
run_eval blank_image "${BLANK_DATA}"

python src/eval/compare_ablation_metrics.py \
  --normal outputs/eval_results/intelli_normal_metrics_by_source.json \
  --text_only outputs/eval_results/intelli_text_only_metrics_by_source.json \
  --wrong_image outputs/eval_results/intelli_wrong_image_metrics_by_source.json \
  --blank_image outputs/eval_results/intelli_blank_image_metrics_by_source.json \
  --output outputs/eval_results/intelli_visual_ablation_summary.json

bash scripts/19_analyze_intelli_ablation_breakdown.sh
