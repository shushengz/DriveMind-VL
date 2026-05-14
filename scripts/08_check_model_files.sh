#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-}"
if [[ -z "$MODEL_DIR" ]]; then
  echo "usage: bash scripts/08_check_model_files.sh /path/to/Qwen2.5-VL-3B-Instruct" >&2
  exit 2
fi

echo "== Model directory =="
echo "$MODEL_DIR"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "missing directory: $MODEL_DIR" >&2
  exit 1
fi

echo
echo "== Required config files =="
for file in config.json preprocessor_config.json tokenizer_config.json; do
  if [[ -f "$MODEL_DIR/$file" ]]; then
    echo "ok: $file"
  else
    echo "missing: $file"
  fi
done

echo
echo "== Weight files =="
find "$MODEL_DIR" -maxdepth 1 \( -name "*.safetensors" -o -name "*.bin" \) -printf "%f %s bytes\n" | sort || true

echo
echo "== Size =="
du -sh "$MODEL_DIR" || true

echo
echo "== Transformers processor sanity check =="
python - "$MODEL_DIR" <<'PY'
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
try:
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    print("processor:", type(processor).__name__)
except Exception as exc:
    print("processor check failed:", exc)
    raise SystemExit(1)
PY

