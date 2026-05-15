#!/usr/bin/env bash
set -euo pipefail

PREDICTIONS=${1:-outputs/eval_results/base_predictions.jsonl}
METRICS_OUTPUT=${2:-outputs/eval_results/metrics_by_source.json}
ERROR_OUTPUT=${3:-outputs/eval_results/error_analysis.json}

python src/eval/eval_by_source.py \
  --predictions "${PREDICTIONS}" \
  --output "${METRICS_OUTPUT}" \
  --bad_cases_output outputs/cases/bad_cases_by_source.jsonl

python src/eval/error_analysis.py \
  --predictions "${PREDICTIONS}" \
  --output "${ERROR_OUTPUT}" \
  --cases_output outputs/cases/error_analysis_cases.jsonl
