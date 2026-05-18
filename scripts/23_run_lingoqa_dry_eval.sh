#!/usr/bin/env bash
set -euo pipefail

INPUT=${INPUT:-data/processed/drivemind_lingoqa_eval_subset.jsonl}
PREDICTIONS=${PREDICTIONS:-outputs/eval_results/lingoqa_dry_predictions.jsonl}
METRICS=${METRICS:-outputs/eval_results/lingoqa_dry_metrics.json}
BREAKDOWN=${BREAKDOWN:-outputs/eval_results/lingoqa_dry_breakdown.json}
BAD_CASES=${BAD_CASES:-outputs/cases/lingoqa_dry_bad_cases.jsonl}

python src/eval/base_infer_dryrun.py \
  --input "${INPUT}" \
  --output "${PREDICTIONS}" \
  --dry_run

python src/eval/run_all_eval.py \
  --predictions "${PREDICTIONS}" \
  --output "${METRICS}" \
  --bad_cases_output "${BAD_CASES}"

python src/eval/eval_external_vqa_breakdown.py \
  --predictions "${PREDICTIONS}" \
  --output "${BREAKDOWN}"
