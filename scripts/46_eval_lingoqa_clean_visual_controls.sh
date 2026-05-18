#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL=${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
ADAPTER=${ADAPTER:-}
SPLIT=${SPLIT:-dev}
DATA_PREFIX=${DATA_PREFIX:-lingoqa_clean_v2}
BASE_DATA=${BASE_DATA:-data/processed/${DATA_PREFIX}_${SPLIT}.jsonl}
WRONG_DATA=${WRONG_DATA:-data/processed/${DATA_PREFIX}_${SPLIT}_wrong_frame.jsonl}
BLANK_DATA=${BLANK_DATA:-data/processed/${DATA_PREFIX}_${SPLIT}_blank_frame.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
MAX_PIXELS=${MAX_PIXELS:-401408}
PROMPT_VARIANT=${PROMPT_VARIANT:-spatial}
FRAME_STRATEGY=${FRAME_STRATEGY:-uniform}
MAX_IMAGES=${MAX_IMAGES:-5}
PREFIX=${PREFIX:-lingoqa_clean_v2_${SPLIT}_qwen25vl_3b}

adapter_args=()
if [[ -n "${ADAPTER}" ]]; then
  adapter_args=(--adapter_path "${ADAPTER}")
fi

run_eval () {
  local tag="$1"
  local input="$2"
  shift 2
  local pred="outputs/eval_results/${tag}_predictions.jsonl"
  local metrics="outputs/eval_results/${tag}_metrics.json"
  local breakdown="outputs/eval_results/${tag}_breakdown.json"
  local bad_cases="outputs/cases/${tag}_bad_cases.jsonl"

  python src/eval/base_infer_qwen25vl.py \
    --input "${input}" \
    --output "${pred}" \
    --model_name_or_path "${MODEL}" \
    "${adapter_args[@]}" \
    --max_samples "${MAX_SAMPLES}" \
    --bf16 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_pixels "${MAX_PIXELS}" \
    --prompt_variant "${PROMPT_VARIANT}" \
    "$@"

  python src/eval/run_all_eval.py \
    --predictions "${pred}" \
    --output "${metrics}" \
    --bad_cases_output "${bad_cases}"

  python src/eval/eval_external_vqa_breakdown.py \
    --predictions "${pred}" \
    --output "${breakdown}"
}

run_eval "${PREFIX}_text_only" "${BASE_DATA}" --text_only
run_eval "${PREFIX}_normal" "${BASE_DATA}" --use_all_images --max_images "${MAX_IMAGES}" --frame_strategy "${FRAME_STRATEGY}"
run_eval "${PREFIX}_wrong" "${WRONG_DATA}" --use_all_images --max_images "${MAX_IMAGES}" --frame_strategy "${FRAME_STRATEGY}"
run_eval "${PREFIX}_blank" "${BLANK_DATA}" --use_all_images --max_images "${MAX_IMAGES}" --frame_strategy "${FRAME_STRATEGY}"

python src/eval/compare_ablation_metrics.py \
  --normal "outputs/eval_results/${PREFIX}_normal_metrics.json" \
  --text_only "outputs/eval_results/${PREFIX}_text_only_metrics.json" \
  --wrong_image "outputs/eval_results/${PREFIX}_wrong_metrics.json" \
  --blank_image "outputs/eval_results/${PREFIX}_blank_metrics.json" \
  --output "outputs/eval_results/${PREFIX}_visual_control_summary.json"

python src/eval/compare_visual_ablation_by_case.py \
  --normal "outputs/eval_results/${PREFIX}_normal_predictions.jsonl" \
  --text_only "outputs/eval_results/${PREFIX}_text_only_predictions.jsonl" \
  --wrong_image "outputs/eval_results/${PREFIX}_wrong_predictions.jsonl" \
  --blank_image "outputs/eval_results/${PREFIX}_blank_predictions.jsonl" \
  --output "outputs/eval_results/${PREFIX}_visual_control_case_summary.json" \
  --cases_output "outputs/cases/${PREFIX}_visual_control_cases.jsonl"
