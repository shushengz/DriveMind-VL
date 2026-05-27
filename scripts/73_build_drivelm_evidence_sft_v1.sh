#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

python src/data/build_drivelm_evidence_sft_v1.py \
  --input "${INPUT:-data/processed/drivelm_train_scene.jsonl}" \
  --output "${OUTPUT:-data/processed/drivelm_evidence_sft_v1_train.jsonl}" \
  --summary "${SUMMARY:-outputs/eval_results/drivelm_evidence_sft_v1_train_summary.json}" \
  --max_samples "${MAX_SAMPLES:-0}" \
  --max_objects_in_reason "${MAX_OBJECTS_IN_REASON:-4}" \
  --seed "${SEED:-20260521}"
