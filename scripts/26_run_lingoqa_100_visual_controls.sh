#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
SOURCE=${SOURCE:-data/external/lingoqa/val.parquet}
IMAGE_ROOT=${IMAGE_ROOT:-data/external/lingoqa}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
MAX_PIXELS=${MAX_PIXELS:-200704}

BASE_DATA=${BASE_DATA:-data/processed/drivemind_lingoqa_eval_100_control.jsonl}
WRONG_DATA=${WRONG_DATA:-data/processed/drivemind_lingoqa_eval_100_wrong_frame.jsonl}
BLANK_DATA=${BLANK_DATA:-data/processed/drivemind_lingoqa_eval_100_blank_frame.jsonl}
BLANK_IMAGE=${BLANK_IMAGE:-outputs/cases/ablation_images/lingoqa_blank.jpg}

IMAGE_ROOT="${IMAGE_ROOT}" TARGET_SIZE=100 OUTPUT="${BASE_DATA}" \
  bash scripts/21_prepare_lingoqa_subset.sh "${SOURCE}"

python src/data/make_visual_ablation.py \
  --input "${BASE_DATA}" \
  --output "${WRONG_DATA}" \
  --mode wrong_image

python src/data/make_visual_ablation.py \
  --input "${BASE_DATA}" \
  --output "${BLANK_DATA}" \
  --mode blank_image \
  --blank_image_path "${BLANK_IMAGE}"

python src/data/validate_dataset.py --input "${WRONG_DATA}"
python src/data/validate_dataset.py --input "${BLANK_DATA}"

run_eval () {
  local tag="$1"
  local input="$2"
  local extra_args="$3"
  local pred="outputs/eval_results/${tag}_predictions.jsonl"
  local metrics="outputs/eval_results/${tag}_metrics.json"
  local breakdown="outputs/eval_results/${tag}_breakdown.json"
  local bad_cases="outputs/cases/${tag}_bad_cases.jsonl"

  python src/eval/base_infer_qwen25vl.py \
    --input "${input}" \
    --output "${pred}" \
    --model_name_or_path "${MODEL}" \
    --max_samples 100 \
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

run_eval lingoqa_qwen25vl_3b_100_control_text_only "${BASE_DATA}" "--text_only"
run_eval lingoqa_qwen25vl_3b_100_control_normal_5frame "${BASE_DATA}" "--use_all_images --max_images 5"
run_eval lingoqa_qwen25vl_3b_100_control_wrong_5frame "${WRONG_DATA}" "--use_all_images --max_images 5"
run_eval lingoqa_qwen25vl_3b_100_control_blank_5frame "${BLANK_DATA}" "--use_all_images --max_images 5"

python src/eval/compare_ablation_metrics.py \
  --normal outputs/eval_results/lingoqa_qwen25vl_3b_100_control_normal_5frame_metrics.json \
  --text_only outputs/eval_results/lingoqa_qwen25vl_3b_100_control_text_only_metrics.json \
  --wrong_image outputs/eval_results/lingoqa_qwen25vl_3b_100_control_wrong_5frame_metrics.json \
  --blank_image outputs/eval_results/lingoqa_qwen25vl_3b_100_control_blank_5frame_metrics.json \
  --output outputs/eval_results/lingoqa_qwen25vl_3b_100_visual_control_summary.json

python src/eval/compare_visual_ablation_by_case.py \
  --normal outputs/eval_results/lingoqa_qwen25vl_3b_100_control_normal_5frame_predictions.jsonl \
  --text_only outputs/eval_results/lingoqa_qwen25vl_3b_100_control_text_only_predictions.jsonl \
  --wrong_image outputs/eval_results/lingoqa_qwen25vl_3b_100_control_wrong_5frame_predictions.jsonl \
  --blank_image outputs/eval_results/lingoqa_qwen25vl_3b_100_control_blank_5frame_predictions.jsonl \
  --output outputs/eval_results/lingoqa_qwen25vl_3b_100_visual_control_case_summary.json \
  --cases_output outputs/cases/lingoqa_qwen25vl_3b_100_visual_control_cases.jsonl
