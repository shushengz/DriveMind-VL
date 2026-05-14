#!/usr/bin/env bash
set -euo pipefail
python src/eval/run_all_eval.py --predictions outputs/eval_results/base_predictions.jsonl --output outputs/eval_results/metrics.json

