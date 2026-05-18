#!/usr/bin/env bash
set -euo pipefail

python src/data/build_lingoqa_preference_v3_control_heavy.py \
  --normal "${NORMAL:-data/processed/lingoqa_clean_v2_train.jsonl}" \
  --wrong "${WRONG:-data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl}" \
  --blank "${BLANK:-data/processed/lingoqa_clean_v2_train_blank_frame.jsonl}" \
  --output "${OUTPUT:-data/processed/lingoqa_preference_v3_control_heavy_train.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/lingoqa_preference_v3_control_heavy_train_summary.json}" \
  --control_repeat "${CONTROL_REPEAT:-3}" \
  --normal_keep_ratio "${NORMAL_KEEP_RATIO:-0.25}" \
  --seed "${SEED:-20260517}"
