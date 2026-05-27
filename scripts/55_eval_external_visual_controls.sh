#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
ADAPTER="${ADAPTER:-}"
DATA="${DATA:-data/processed/external_eval.jsonl}"
PREFIX="${PREFIX:-external_eval_qwen25vl}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_PIXELS="${MAX_PIXELS:-401408}"
PROMPT_VARIANT="${PROMPT_VARIANT:-spatial}"
PERCEPTION_MODE="${PERCEPTION_MODE:-full}"
FRAME_STRATEGY="${FRAME_STRATEGY:-uniform}"
MAX_IMAGES="${MAX_IMAGES:-5}"
DRY_RUN="${DRY_RUN:-0}"
WRONG_STRATEGY="${WRONG_STRATEGY:-hard_scene}"
ABLATION_SEED="${ABLATION_SEED:-20260521}"

WRONG_DATA="${WRONG_DATA:-${DATA%.jsonl}_wrong_image.jsonl}"
BLANK_DATA="${BLANK_DATA:-${DATA%.jsonl}_blank_image.jsonl}"
BLANK_IMAGE_PATH="${BLANK_IMAGE_PATH:-outputs/cases/ablation_images/${PREFIX}_blank.jpg}"

mkdir -p outputs/eval_results outputs/cases outputs/logs

python src/data/make_visual_ablation.py \
  --input "${DATA}" \
  --output "${WRONG_DATA}" \
  --mode wrong_image \
  --blank_image_path "${BLANK_IMAGE_PATH}" \
  --wrong_strategy "${WRONG_STRATEGY}" \
  --seed "${ABLATION_SEED}"

python src/data/make_visual_ablation.py \
  --input "${DATA}" \
  --output "${BLANK_DATA}" \
  --mode blank_image \
  --blank_image_path "${BLANK_IMAGE_PATH}"

adapter_args=()
if [[ -n "${ADAPTER}" ]]; then
  adapter_args=(--adapter_path "${ADAPTER}")
fi

dry_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
  dry_args=(--dry_run)
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
    "${dry_args[@]}" \
    --max_samples "${MAX_SAMPLES}" \
    --bf16 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_pixels "${MAX_PIXELS}" \
    --prompt_variant "${PROMPT_VARIANT}" \
    --perception_mode "${PERCEPTION_MODE}" \
    "$@"

  python src/eval/run_all_eval.py \
    --predictions "${pred}" \
    --output "${metrics}" \
    --bad_cases_output "${bad_cases}"

  python src/eval/eval_external_vqa_breakdown.py \
    --predictions "${pred}" \
    --output "${breakdown}"
}

run_eval "${PREFIX}_text_only" "${DATA}" --text_only
run_eval "${PREFIX}_normal" "${DATA}" --use_all_images --max_images "${MAX_IMAGES}" --frame_strategy "${FRAME_STRATEGY}"
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
