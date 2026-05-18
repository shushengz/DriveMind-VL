#!/usr/bin/env bash
set -euo pipefail

python src/data/build_sft_v0_from_review.py \
  --review_csv outputs/cases/lingoqa_sft_candidate_review_agent_checked.csv \
  --candidate_jsonl data/processed/lingoqa_sft_candidates_needs_review.jsonl \
  --keep_output data/processed/lingoqa_sft_v0_keep_25.jsonl \
  --fix_output outputs/cases/lingoqa_sft_v0_need_fix_7.csv \
  --drop_output outputs/cases/lingoqa_sft_v0_drop_9.csv \
  --report_output docs/lingoqa_sft_v0_build_report.md \
  --note_output Note/2026-05-17_LingoQA_SFT_v0数据分流.md
