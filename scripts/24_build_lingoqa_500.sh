#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:-data/external/lingoqa/val.parquet}
OUTPUT=${OUTPUT:-data/processed/drivemind_lingoqa_eval_500.jsonl}
IMAGE_ROOT=${IMAGE_ROOT:-data/external/lingoqa}

IMAGE_ROOT="${IMAGE_ROOT}" TARGET_SIZE=500 OUTPUT="${OUTPUT}" \
  bash scripts/21_prepare_lingoqa_subset.sh "${INPUT}"
