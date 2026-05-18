#!/usr/bin/env bash
set -euo pipefail

python src/data/build_lingoqa_sft_candidates.py \
  --cases outputs/cases/lingoqa_qwen25vl_3b_100_best_spatial_3frame_visual_control_cases.jsonl \
  --dataset data/processed/drivemind_lingoqa_eval_100_control.jsonl \
  --output_jsonl data/processed/lingoqa_sft_candidates_needs_review.jsonl \
  --review_csv outputs/cases/lingoqa_sft_candidate_review.csv \
  --summary outputs/eval_results/lingoqa_sft_candidate_summary.json \
  --report docs/lingoqa_sft_candidate_curation.md
