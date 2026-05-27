#!/usr/bin/env bash
set -euo pipefail
case "${OMP_NUM_THREADS:-}" in ""|0|*[!0-9]*) export OMP_NUM_THREADS=1 ;; esac
case "${MKL_NUM_THREADS:-}" in ""|0|*[!0-9]*) export MKL_NUM_THREADS="$OMP_NUM_THREADS" ;; esac

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gpu_stage}"
MAX_CHECK_ROWS="${MAX_CHECK_ROWS:-20}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --max_check_rows) MAX_CHECK_ROWS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
LOG_PATH="$OUTPUT_DIR/check_gpu_stage_ready.log"
JSON_PATH="$OUTPUT_DIR/check_gpu_stage_ready.json"

{
  echo "[check] python=$PYTHON_BIN"
  echo "[check] model_path=$MODEL_PATH"
  "$PYTHON_BIN" - "$MODEL_PATH" "$JSON_PATH" "$MAX_CHECK_ROWS" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
json_path = Path(sys.argv[2])
max_rows = int(sys.argv[3])
root = Path.cwd()
errors = []
warnings = []
checks = {}

def add(name, ok, detail=""):
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        errors.append(f"{name}: {detail}")

def warn(name, detail):
    warnings.append(f"{name}: {detail}")

add("python_executable", True, sys.executable)
try:
    import torch
    checks["torch"] = {"ok": True, "version": torch.__version__}
    cuda_ok = torch.cuda.is_available()
    checks["cuda"] = {"ok": bool(cuda_ok), "device_count": torch.cuda.device_count() if cuda_ok else 0}
    if not cuda_ok:
        errors.append("cuda: torch.cuda.is_available() is false")
except Exception as exc:
    add("torch", False, repr(exc))
    checks["cuda"] = {"ok": False, "detail": "torch import failed"}

for pkg in ["transformers", "peft", "accelerate", "bitsandbytes", "qwen_vl_utils"]:
    spec = importlib.util.find_spec(pkg)
    checks[pkg] = {"ok": spec is not None}
    if spec is None:
        errors.append(f"package {pkg} is not importable")

add("model_path", model_path.exists(), model_path.as_posix())
if model_path.exists() and importlib.util.find_spec("transformers") is not None:
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path.as_posix(), trust_remote_code=True, use_fast=False)
        checks["qwen25vl_processor"] = {"ok": True, "class": proc.__class__.__name__}
    except Exception as exc:
        add("qwen25vl_processor", False, repr(exc))
else:
    add("qwen25vl_processor", False, "model path missing or transformers unavailable")

for directory in ["data/processed/visual_control", "data/train/sft_v3", "data/train/preference_v8"]:
    p = root / directory
    add(directory, p.exists() and p.is_dir(), p.as_posix())

blank_candidates = [root / "outputs/cases/ablation_images/blank.jpg", root / "outputs/cases/ablation_images/lingoqa_blank.jpg"]
checks["blank_image_placeholder"] = {"ok": any(p.exists() for p in blank_candidates), "candidates": [p.as_posix() for p in blank_candidates]}
if not checks["blank_image_placeholder"]["ok"]:
    errors.append("blank image placeholder not found")

required = ["id", "dataset", "mode", "setting", "question", "gold", "image_paths", "image_labels", "prompt", "metadata"]
leak_keys = {"perception", "objects", "raw_qa", "key_object_infos"}

def contains_leak_key(value):
    if isinstance(value, dict):
        return any(str(key) in leak_keys or contains_leak_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_leak_key(item) for item in value)
    return False

sample_report = {}
for dataset in ["lingoqa", "drivelm"]:
    for setting in ["normal", "text_only", "wrong_image", "blank_image"]:
        key = f"{dataset}_{setting}"
        path = root / f"data/processed/visual_control/{dataset}_strict_{setting}.jsonl"
        if not path.exists():
            add(f"visual_file_{key}", False, path.as_posix())
            continue
        total = 0
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
                    if len(rows) < max_rows:
                        rows.append(json.loads(line))
        sample_report[key] = {"path": path.as_posix(), "checked_rows": len(rows), "total_rows": total}
        if total < 50:
            warn(key, f"only {total} rows; run mode should rebuild full visual-control data before real eval")
        for idx, row in enumerate(rows):
            missing = [name for name in required if name not in row]
            if missing:
                errors.append(f"{key}:{idx} missing fields {missing}")
            if row.get("mode") != "strict_visual":
                errors.append(f"{key}:{idx} mode is not strict_visual")
            if contains_leak_key(row):
                errors.append(f"{key}:{idx} possible strict leakage key")
            image_paths = row.get("image_paths") if isinstance(row.get("image_paths"), list) else []
            if setting == "text_only" and image_paths:
                errors.append(f"{key}:{idx} text_only image_paths is not empty")
            if setting != "text_only" and not image_paths:
                errors.append(f"{key}:{idx} non-text setting has empty image_paths")
            for item in image_paths[:6]:
                item_path = root / str(item)
                if not item_path.exists():
                    if setting == "blank_image" and any(p.exists() for p in blank_candidates):
                        warn(key, f"blank path {item} missing; generic blank placeholder exists")
                    else:
                        errors.append(f"{key}:{idx} image path missing: {item}")
checks["visual_control_samples"] = sample_report
result = {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}
json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    sys.exit(1)
PY
} 2>&1 | tee "$LOG_PATH"
