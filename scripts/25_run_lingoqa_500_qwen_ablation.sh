#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
INPUT=${INPUT:-data/processed/drivemind_lingoqa_eval_500.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-500}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
MAX_PIXELS=${MAX_PIXELS:-200704}

run_eval () {
  local tag="$1"
  local extra_args="$2"
  local pred="outputs/eval_results/${tag}_predictions.jsonl"
  local metrics="outputs/eval_results/${tag}_metrics.json"
  local breakdown="outputs/eval_results/${tag}_breakdown.json"
  local bad_cases="outputs/cases/${tag}_bad_cases.jsonl"

  python src/eval/base_infer_qwen25vl.py \
    --input "${INPUT}" \
    --output "${pred}" \
    --model_name_or_path "${MODEL}" \
    --max_samples "${MAX_SAMPLES}" \
    --bf16 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_pixels "${MAX_PIXELS}" \
    ${extra_args}

  python src/eval/run_all_eval.py \
    --predictions "${pred}" \
    --output "${metrics}" \
    --bad_cases_output "${bad_cases}"

  python src/eval/eval_external_vqa_breakdown.py \
    --predictions "${pred}" \
    --output "${breakdown}"
}

run_eval lingoqa_qwen25vl_3b_500_text_only "--text_only"
run_eval lingoqa_qwen25vl_3b_500_single_frame ""
run_eval lingoqa_qwen25vl_3b_500_5frame "--use_all_images --max_images 5"
