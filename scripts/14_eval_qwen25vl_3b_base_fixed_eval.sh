#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

python src/eval/base_infer_qwen25vl.py \
  --input data/processed/drivemind_eval.jsonl \
  --output outputs/eval_results/qwen25vl_3b_base_eval_predictions.jsonl \
  --model_name_or_path "$MODEL_DIR" \
  --max_samples "${MAX_SAMPLES:-20}" \
  --bf16

python src/eval/run_all_eval.py \
  --predictions outputs/eval_results/qwen25vl_3b_base_eval_predictions.jsonl \
  --output outputs/eval_results/qwen25vl_3b_base_eval_metrics.json \
  --bad_cases_output outputs/cases/qwen25vl_3b_base_eval_bad_cases.jsonl

