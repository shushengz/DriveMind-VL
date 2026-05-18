#!/usr/bin/env bash
set -euo pipefail

INPUT_JSONL=${1:-third_party/IntelliCockpitBench/Evaluation/data/jsonl/english_test.jsonl}
IMAGE_ROOT=${2:-third_party/IntelliCockpitBench/Evaluation/data/images}
TARGET_SIZE=${TARGET_SIZE:-100}
SEED=${SEED:-42}
OUTPUT=${OUTPUT:-data/processed/drivemind_intelli_eval_subset.jsonl}

python src/data/audit_intelli_cockpitbench.py \
  --input "${INPUT_JSONL}" \
  --image_root "${IMAGE_ROOT}" \
  --output outputs/eval_results/intelli_dataset_audit.json

python src/data/build_intelli_eval_subset.py \
  --input "${INPUT_JSONL}" \
  --image_root "${IMAGE_ROOT}" \
  --target_size "${TARGET_SIZE}" \
  --seed "${SEED}" \
  --output "${OUTPUT}" \
  --manifest_output outputs/eval_results/intelli_eval_subset_manifest.json

python src/data/validate_dataset.py \
  --input "${OUTPUT}"
