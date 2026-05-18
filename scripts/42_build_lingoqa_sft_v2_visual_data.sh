#!/usr/bin/env bash
set -euo pipefail

python src/data/build_lingoqa_sft_v2_visual.py \
  --input "${INPUT:-data/processed/lingoqa_clean_v2_train.jsonl}" \
  --output "${OUTPUT:-data/processed/lingoqa_sft_v2_visual_train.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/lingoqa_sft_v2_visual_train_summary.json}" \
  --reference_variants "${REFERENCE_VARIANTS:-all}" \
  --capability_repeats "${CAPABILITY_REPEATS:-}" \
  --seed "${SEED:-20260517}"
