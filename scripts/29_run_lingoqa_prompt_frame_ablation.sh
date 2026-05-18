#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
DATA=${DATA:-data/processed/drivemind_lingoqa_eval_100_control.jsonl}
MAX_SAMPLES=${MAX_SAMPLES:-100}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
MAX_PIXELS=${MAX_PIXELS:-200704}

if [[ ! -f "${DATA}" ]]; then
  IMAGE_ROOT=${IMAGE_ROOT:-data/external/lingoqa} TARGET_SIZE="${MAX_SAMPLES}" OUTPUT="${DATA}" \
    bash scripts/21_prepare_lingoqa_subset.sh "${SOURCE:-data/external/lingoqa/val.parquet}"
fi

run_one () {
  local prompt_variant="$1"
  local frame_strategy="$2"
  local max_images="$3"
  local tag="lingoqa_qwen25vl_3b_100_prompt-${prompt_variant}_frame-${frame_strategy}_${max_images}img"
  local pred="outputs/eval_results/${tag}_predictions.jsonl"
  local metrics="outputs/eval_results/${tag}_metrics.json"
  local breakdown="outputs/eval_results/${tag}_breakdown.json"
  local bad_cases="outputs/cases/${tag}_bad_cases.jsonl"

  python src/eval/base_infer_qwen25vl.py \
    --input "${DATA}" \
    --output "${pred}" \
    --model_name_or_path "${MODEL}" \
    --max_samples "${MAX_SAMPLES}" \
    --bf16 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --max_pixels "${MAX_PIXELS}" \
    --use_all_images \
    --max_images "${max_images}" \
    --frame_strategy "${frame_strategy}" \
    --prompt_variant "${prompt_variant}"

  python src/eval/run_all_eval.py \
    --predictions "${pred}" \
    --output "${metrics}" \
    --bad_cases_output "${bad_cases}"

  python src/eval/eval_external_vqa_breakdown.py \
    --predictions "${pred}" \
    --output "${breakdown}"
}

# Keep this matrix intentionally small so it can run on a 32GB GPU smoke server.
run_one current first_n 5
run_one temporal uniform 5
run_one spatial uniform 5
run_one evidence uniform 5
run_one spatial first_middle_last 3

python src/eval/compare_prompt_frame_ablation.py \
  --metrics_glob "outputs/eval_results/lingoqa_qwen25vl_3b_100_prompt-*_metrics.json" \
  --breakdown_glob "outputs/eval_results/lingoqa_qwen25vl_3b_100_prompt-*_breakdown.json" \
  --output outputs/eval_results/lingoqa_prompt_frame_ablation_summary.json \
  --report_output docs/lingoqa_prompt_frame_ablation_report.md
