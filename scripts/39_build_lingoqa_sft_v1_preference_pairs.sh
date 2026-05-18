#!/usr/bin/env bash
set -euo pipefail

python src/data/build_lingoqa_sft_v1_preference_pairs.py \
  --input "${INPUT:-data/processed/lingoqa_sft_v1_control_aware_77.jsonl}" \
  --original_lookup "${ORIGINAL_LOOKUP:-data/processed/lingoqa_sft_v1_balanced_67.jsonl}" \
  --output "${OUTPUT:-data/processed/lingoqa_sft_v1_preference_pairs_77.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/lingoqa_sft_v1_preference_pairs_77_summary.json}"
