#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
EVAL_IDS="${2:-outputs/final_report/drivelm_ood_ids_100.json}"
OUT_DIR="outputs/gpu_stage"
OUT_JSON="$OUT_DIR/stage13_drivelm_ood_ready.json"
OUT_LOG="$OUT_DIR/stage13_drivelm_ood_ready.log"
mkdir -p "$OUT_DIR"

set +e
"$PYTHON_BIN" - "$MODEL_PATH" "$EVAL_IDS" "$OUT_JSON" 2>&1 <<'PY' | tee "$OUT_LOG"
from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
eval_ids_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
settings = ("normal", "text_only", "wrong_image", "blank_image")
pool_dir = Path("data/processed/visual_control")
errors: list[str] = []
warnings: list[str] = []
checks: dict[str, object] = {}

try:
    import torch
    checks["cuda_available"] = bool(torch.cuda.is_available())
    checks["cuda_device_count"] = int(torch.cuda.device_count())
    checks["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    if not checks["cuda_available"]:
        errors.append("CUDA is unavailable")
except Exception as exc:
    checks["cuda_available"] = False
    errors.append(f"cannot import torch or inspect CUDA: {exc}")

for package in ("transformers", "peft", "accelerate", "bitsandbytes"):
    try:
        module = importlib.import_module(package)
        checks[f"{package}_available"] = True
        checks[f"{package}_version"] = getattr(module, "__version__", "")
    except Exception as exc:
        checks[f"{package}_available"] = False
        errors.append(f"cannot import {package}: {exc}")

checks["model_path"] = model_path.as_posix()
checks["model_path_exists"] = model_path.exists()
if model_path.exists():
    try:
        from transformers import AutoConfig, AutoProcessor
        AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        checks["local_model_config_processor_loadable"] = True
    except Exception as exc:
        checks["local_model_config_processor_loadable"] = False
        errors.append(f"local model config/processor cannot load: {exc}")
else:
    checks["local_model_config_processor_loadable"] = False
    errors.append(f"model path missing: {model_path}")

adapter = Path("checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke")
checks["r3_adapter_exists"] = adapter.exists() and (adapter / "adapter_model.safetensors").exists()
if not checks["r3_adapter_exists"]:
    errors.append("SFT-v3-r3 adapter is missing")

required = [
    Path("outputs/final_report/drivelm_ood_ids_100.json"),
    Path("outputs/final_report/drivelm_ood_ids_300.json"),
    Path("outputs/final_report/drivelm_ood_pool_summary.json"),
] + [pool_dir / f"drivelm_strict_{setting}.jsonl" for setting in settings]
missing = [path.as_posix() for path in required if not path.exists()]
checks["required_files_present"] = not missing
checks["missing_files"] = missing
if missing:
    errors.append("missing required files: " + ", ".join(missing))

def rows(path: Path) -> dict[str, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[str(row.get("id"))] = row
    return result

requested_ids: list[str] = []
maps: dict[str, dict[str, dict]] = {}
if not missing:
    payload = json.loads(eval_ids_path.read_text(encoding="utf-8"))
    requested_ids = payload.get("ids", payload) if isinstance(payload, dict) else payload
    requested_ids = [str(item) for item in requested_ids]
    checks["selected_eval_ids"] = len(requested_ids)
    if not requested_ids or len(requested_ids) != len(set(requested_ids)):
        errors.append("eval IDs are empty or duplicated")
    maps = {setting: rows(pool_dir / f"drivelm_strict_{setting}.jsonl") for setting in settings}
    missing_selected = {
        setting: [sample_id for sample_id in requested_ids if sample_id not in maps[setting]]
        for setting in settings
    }
    checks["eval_ids_aligned"] = not any(missing_selected.values())
    if not checks["eval_ids_aligned"]:
        errors.append("selected eval IDs are missing from one or more settings")

perception_re = re.compile(r'"(?:objects?|perception|bbox|coordinates?|detections?)"\s*:', re.I)
violations = {
    "text_only_has_images": [],
    "wrong_image_equals_normal": [],
    "blank_image_not_placeholder": [],
    "non_strict_mode": [],
    "perception_json_leak": [],
}
if requested_ids and maps and checks.get("eval_ids_aligned"):
    for sample_id in requested_ids:
        group = {setting: maps[setting][sample_id] for setting in settings}
        if group["text_only"].get("image_paths"):
            violations["text_only_has_images"].append(sample_id)
        if group["wrong_image"].get("image_paths") == group["normal"].get("image_paths"):
            violations["wrong_image_equals_normal"].append(sample_id)
        blank_paths = [str(path).lower() for path in group["blank_image"].get("image_paths", [])]
        if not blank_paths or not all(any(word in path for word in ("blank", "placeholder", "empty")) for path in blank_paths):
            violations["blank_image_not_placeholder"].append(sample_id)
        if any(group[setting].get("mode") != "strict_visual" for setting in settings):
            violations["non_strict_mode"].append(sample_id)
        exposed = "\n".join(
            str(group[setting].get(field, ""))
            for setting in settings
            for field in ("question", "prompt", "gold")
        )
        if perception_re.search(exposed):
            violations["perception_json_leak"].append(sample_id)
checks["violations"] = {key: len(value) for key, value in violations.items()}
for key, value in violations.items():
    if value:
        errors.append(f"{key}: {len(value)} selected cases, examples={value[:3]}")

summary = Path("outputs/final_report/drivelm_ood_pool_summary.json")
if summary.exists():
    pool_summary = json.loads(summary.read_text(encoding="utf-8"))
    checks["pool_four_setting_complete"] = bool(pool_summary.get("four_setting_complete"))
    checks["pool_eligible_ood_ids"] = int(pool_summary.get("eligible_ood_ids", 0))
    if not checks["pool_four_setting_complete"]:
        errors.append("DriveLM OOD pool summary is not four-setting complete")

report = {
    "stage": "stage13_drivelm_ood_strict_visual_gpu_eval",
    "ready": not errors,
    "eval_ids_path": eval_ids_path.as_posix(),
    "checks": checks,
    "errors": errors,
    "warnings": warnings,
    "constraints": {
        "inference_only": True,
        "training_forbidden": True,
        "grpo_forbidden": True,
        "lingoqa_prediction_overwrite_forbidden": True,
        "drivelm_training_data_construction_forbidden": True,
    },
}
out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
raise SystemExit(0 if report["ready"] else 1)
PY
status=${PIPESTATUS[0]}
set -e
exit "$status"
