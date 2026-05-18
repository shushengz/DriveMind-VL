#!/usr/bin/env bash
set -euo pipefail

python src/data/build_lingoqa_clean_splits.py \
  --input "${INPUT:-data/processed/drivemind_lingoqa_eval_500.jsonl}" \
  --prefix "${PREFIX:-lingoqa_clean_v2}" \
  --train_ratio "${TRAIN_RATIO:-0.6}" \
  --dev_ratio "${DEV_RATIO:-0.2}" \
  --seed "${SEED:-20260517}"
