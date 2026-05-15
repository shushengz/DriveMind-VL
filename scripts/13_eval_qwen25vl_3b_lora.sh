#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
ADAPTER_DIR="${2:-outputs/checkpoints/qwen25vl_3b_lora_smoke}"

python src/eval/base_infer_qwen25vl.py \
  --input data/processed/drivemind_eval.jsonl \
  --output outputs/eval_results/qwen25vl_3b_lora_predictions.jsonl \
  --model_name_or_path "$MODEL_DIR" \
  --adapter_path "$ADAPTER_DIR" \
  --max_samples "${MAX_SAMPLES:-20}" \
  --bf16

python src/eval/run_all_eval.py \
  --predictions outputs/eval_results/qwen25vl_3b_lora_predictions.jsonl \
  --output outputs/eval_results/qwen25vl_3b_lora_metrics.json \
  --bad_cases_output outputs/cases/qwen25vl_3b_lora_bad_cases.jsonl

