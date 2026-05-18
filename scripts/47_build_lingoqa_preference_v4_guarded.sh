#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python src/data/build_lingoqa_preference_v4_guarded.py \
  --normal "${NORMAL:-data/processed/lingoqa_clean_v2_train.jsonl}" \
  --wrong "${WRONG:-data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl}" \
  --blank "${BLANK:-data/processed/lingoqa_clean_v2_train_blank_frame.jsonl}" \
  --output "${OUTPUT:-data/processed/lingoqa_preference_v4_guarded_train.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/lingoqa_preference_v4_guarded_train_summary.json}" \
  --wrong_repeat "${WRONG_REPEAT:-2}" \
  --blank_repeat "${BLANK_REPEAT:-1}" \
  --text_repeat "${TEXT_REPEAT:-1}" \
  --seed "${SEED:-20260517}"
