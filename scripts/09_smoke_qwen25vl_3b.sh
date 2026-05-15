#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/mnt/models/Qwen2.5-VL-3B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"

bash scripts/08_check_model_files.sh "$MODEL_DIR"

python src/eval/base_infer_qwen25vl.py \
  --input data/processed/drivemind_seed.jsonl \
  --output outputs/eval_results/qwen25vl_3b_smoke_predictions.jsonl \
  --model_name_or_path "$MODEL_DIR" \
  --max_samples "$MAX_SAMPLES" \
  --bf16

python src/eval/run_all_eval.py \
  --predictions outputs/eval_results/qwen25vl_3b_smoke_predictions.jsonl \
  --output outputs/eval_results/qwen25vl_3b_smoke_metrics.json \
  --bad_cases_output outputs/cases/qwen25vl_3b_smoke_bad_cases.jsonl
