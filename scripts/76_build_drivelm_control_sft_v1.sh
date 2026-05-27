#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python src/data/build_drivelm_control_sft_v1.py \
  --normal_data "${NORMAL_DATA:-data/processed/drivelm_evidence_sft_v1_train.jsonl}" \
  --wrong_data "${WRONG_DATA:-data/processed/drivelm_train_scene_wrong_image_hard.jsonl}" \
  --blank_data "${BLANK_DATA:-data/processed/drivelm_train_scene_blank_image_v4.jsonl}" \
  --output "${OUTPUT:-data/processed/drivelm_control_sft_v1_train.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/drivelm_control_sft_v1_train_summary.json}" \
  --seed "${SEED:-20260522}" \
  --control_modes "${CONTROL_MODES:-text_only,wrong_image,blank_image}" \
  --max_normal_rows "${MAX_NORMAL_ROWS:-1200}" \
  --max_control_rows_per_mode "${MAX_CONTROL_ROWS_PER_MODE:-700}" \
  --max_control_rows_per_capability "${MAX_CONTROL_ROWS_PER_CAPABILITY:-180}"
