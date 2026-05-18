#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS=${1:-outputs/eval_results/lingoqa_qwen25vl_3b_100_control_normal_5frame_predictions.jsonl}
REFERENCES=${REFERENCES:-data/external/lingoqa/val.parquet}
TAG=${TAG:-lingoqa_qwen25vl_3b_100_control_normal_5frame}
MAX_SAMPLES=${MAX_SAMPLES:-100}

CSV="outputs/eval_results/${TAG}_lingoqa_predictions.csv"
OUT="outputs/eval_results/${TAG}_lingo_judge_metrics.json"

python src/eval/export_lingoqa_predictions_csv.py \
  --predictions "${PREDICTIONS}" \
  --output "${CSV}"

python src/eval/eval_lingo_judge.py \
  --predictions_csv "${CSV}" \
  --references "${REFERENCES}" \
  --output "${OUT}" \
  --max_samples "${MAX_SAMPLES}"
