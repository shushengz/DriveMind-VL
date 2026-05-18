#!/usr/bin/env bash
set -euo pipefail

OUTPUT=${OUTPUT:-data/external/lingoqa/evaluation.parquet}

if [[ "${ACCEPT_LINGOQA_TERMS:-}" != "1" ]]; then
  echo "Refusing to download until ACCEPT_LINGOQA_TERMS=1 is set."
  echo "Review upstream repository and license first: https://github.com/wayveai/LingoQA"
  exit 2
fi

python src/data/download_lingoqa_metadata.py \
  --output "${OUTPUT}" \
  --accept_terms
