#!/usr/bin/env bash
set -euo pipefail
python src/eval/base_infer_qwen25vl.py \
  --input data/processed/drivemind_seed.jsonl \
  --output outputs/eval_results/qwen25vl_predictions.jsonl \
  --max_samples 5 \
  "$@"

