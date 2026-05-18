#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-data/external/lingoqa/evaluation.parquet}
TARGET_SIZE=${TARGET_SIZE:-100}
OUTPUT=${OUTPUT:-data/processed/drivemind_lingoqa_eval_subset.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-}
VIDEO_ROOT=${VIDEO_ROOT:-}

ARGS=(
  src/data/prepare_lingoqa_subset.py
  --input "${INPUT}"
  --target_size "${TARGET_SIZE}"
  --output "${OUTPUT}"
  --manifest_output outputs/eval_results/lingoqa_eval_subset_manifest.json
)

if [[ -n "${IMAGE_ROOT}" ]]; then
  ARGS+=(--image_root "${IMAGE_ROOT}")
fi

if [[ -n "${VIDEO_ROOT}" ]]; then
  ARGS+=(--video_root "${VIDEO_ROOT}")
fi

python "${ARGS[@]}"

python src/data/validate_dataset.py \
  --input "${OUTPUT}"
