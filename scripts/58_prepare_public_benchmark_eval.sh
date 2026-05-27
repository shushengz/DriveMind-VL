#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

SOURCE="${SOURCE:-drivelm}"
INPUT="${INPUT:-data/external/${SOURCE}/annotations.json}"
IMAGE_ROOT="${IMAGE_ROOT:-data/external/${SOURCE}}"
SPLIT="${SPLIT:-eval}"
LIMIT="${LIMIT:-0}"
OUTPUT="${OUTPUT:-data/processed/${SOURCE}_${SPLIT}.jsonl}"
AUDIT_JSON="${AUDIT_JSON:-outputs/eval_results/${SOURCE}_${SPLIT}_audit.json}"
AUDIT_MD="${AUDIT_MD:-docs/${SOURCE}_${SPLIT}_audit.md}"

if [[ ! -f "${INPUT}" ]]; then
  echo "ERROR: input annotation file does not exist: ${INPUT}" >&2
  echo "" >&2
  echo "Existing JSON-like files under data/external/${SOURCE}:" >&2
  find "data/external/${SOURCE}" -type f \( -iname '*.json' -o -iname '*.jsonl' \) -print 2>/dev/null | head -50 >&2 || true
  echo "" >&2
  if [[ "${SOURCE}" == "drivelm" ]]; then
    echo "Expected DriveLM placement:" >&2
    echo "  data/external/drivelm/v1_0_train_nus.json" >&2
    echo "  data/external/drivelm/nuscenes/samples/..." >&2
    echo "Official metadata URL:" >&2
    echo "  https://huggingface.co/datasets/OpenDriveLab/DriveLM/resolve/main/v1_0_train_nus.json" >&2
  elif [[ "${SOURCE}" == "nuscenes_qa" ]]; then
    echo "Expected NuScenes-QA placement:" >&2
    echo "  data/external/nuscenes_qa/questions/NuScenes_val_questions.json" >&2
    echo "  data/external/nuscenes_qa/questions/NuScenes_train_questions.json" >&2
  fi
  exit 2
fi

python src/data/convert_external_to_drivemind.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --source "${SOURCE}" \
  --image_root "${IMAGE_ROOT}" \
  --split "${SPLIT}" \
  --limit "${LIMIT}"

python src/data/audit_external_benchmark.py \
  --input "${OUTPUT}" \
  --output "${AUDIT_JSON}" \
  --markdown "${AUDIT_MD}"
