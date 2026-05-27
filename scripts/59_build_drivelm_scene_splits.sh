#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SOURCE="${SOURCE:-drivelm}"
INPUT="${INPUT:-data/external/drivelm/v1_1_train_nus.json}"
IMAGE_ROOT="${IMAGE_ROOT:-data/external/drivelm}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-data/processed/drivelm_train_scene.jsonl}"
DEV_OUTPUT="${DEV_OUTPUT:-data/processed/drivelm_dev_scene.jsonl}"
REPORT_JSON="${REPORT_JSON:-outputs/eval_results/drivelm_scene_split_report.json}"
REPORT_MD="${REPORT_MD:-docs/drivelm_scene_split_report.md}"
TRAIN_LIMIT="${TRAIN_LIMIT:-2400}"
DEV_LIMIT="${DEV_LIMIT:-600}"
DEV_ROWS="${DEV_ROWS:-600}"
MIN_DEV_SCENES="${MIN_DEV_SCENES:-30}"
SEED="${SEED:-42}"

python src/data/build_drivelm_scene_splits.py \
  --input "${INPUT}" \
  --image_root "${IMAGE_ROOT}" \
  --train_output "${TRAIN_OUTPUT}" \
  --dev_output "${DEV_OUTPUT}" \
  --report_json "${REPORT_JSON}" \
  --report_md "${REPORT_MD}" \
  --source "${SOURCE}" \
  --seed "${SEED}" \
  --dev_rows "${DEV_ROWS}" \
  --min_dev_scenes "${MIN_DEV_SCENES}" \
  --train_limit "${TRAIN_LIMIT}" \
  --dev_limit "${DEV_LIMIT}"

python src/data/audit_external_benchmark.py \
  --input "${TRAIN_OUTPUT}" \
  --output "outputs/eval_results/drivelm_train_scene_audit.json" \
  --markdown "docs/drivelm_train_scene_audit.md"

python src/data/audit_external_benchmark.py \
  --input "${DEV_OUTPUT}" \
  --output "outputs/eval_results/drivelm_dev_scene_audit.json" \
  --markdown "docs/drivelm_dev_scene_audit.md"
