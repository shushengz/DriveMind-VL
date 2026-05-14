#!/usr/bin/env bash
set -euo pipefail
python src/eval/base_infer_dryrun.py --input data/processed/drivemind_seed.jsonl --output outputs/eval_results/base_predictions.jsonl --dry_run

