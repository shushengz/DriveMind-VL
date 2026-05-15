#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH=${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}
INTELLI_REPO=${2:-third_party/IntelliCockpitBench}
MAX_SAMPLES=${MAX_SAMPLES:-20}

python src/data/prepare_intelli_cockpitbench_sample.py \
  --repo_root "${INTELLI_REPO}" \
  --output data/processed/drivemind_intelli_sample.jsonl \
  --limit "${MAX_SAMPLES}"

python src/data/validate_dataset.py \
  --input data/processed/drivemind_intelli_sample.jsonl

python src/eval/base_infer_qwen25vl.py \
  --input data/processed/drivemind_intelli_sample.jsonl \
  --output outputs/eval_results/intelli_qwen25vl_3b_base_predictions.jsonl \
  --model_name_or_path "${MODEL_PATH}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_new_tokens 160 \
  --bf16

python src/eval/eval_by_source.py \
  --predictions outputs/eval_results/intelli_qwen25vl_3b_base_predictions.jsonl \
  --output outputs/eval_results/intelli_qwen25vl_3b_base_metrics_by_source.json \
  --bad_cases_output outputs/cases/intelli_qwen25vl_3b_base_bad_cases.jsonl

python src/eval/error_analysis.py \
  --predictions outputs/eval_results/intelli_qwen25vl_3b_base_predictions.jsonl \
  --output outputs/eval_results/intelli_qwen25vl_3b_base_error_analysis.json \
  --cases_output outputs/cases/intelli_qwen25vl_3b_base_error_cases.jsonl
