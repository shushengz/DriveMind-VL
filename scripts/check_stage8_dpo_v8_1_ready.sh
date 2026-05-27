#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
CONFIG_PATH="${CONFIG_PATH:-configs/dpo_v8_1_lingo_smoke.yaml}"
mkdir -p outputs/gpu_stage

STAGE8_MODEL_PATH="$MODEL_PATH" STAGE8_CONFIG_PATH="$CONFIG_PATH" "$PYTHON_BIN" - <<'PY' 2>&1 | tee outputs/gpu_stage/stage8_dpo_v8_1_ready.log
import importlib.util
import json
import os
import sys
from pathlib import Path

model_path = Path(os.environ["STAGE8_MODEL_PATH"])
config_path = Path(os.environ["STAGE8_CONFIG_PATH"])
r3 = "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
preference = Path("data/train/preference_v8_1/preference_v8_1_pairs.jsonl")
audit_path = Path("outputs/data_audit/preference_v8_1_audit.json")
heldout_path = Path("outputs/final_report/stage4_5_heldout_ids_100.json")
heldout_pool = Path("data/processed/visual_control_heldout")
errors, warnings, checks = [], [], {}

def add(name, ok, detail=None):
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        errors.append(f"{name}: {detail}")

def yaml_values(path):
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
    return values

add("python_executable", True, sys.executable)
try:
    import torch
    add("torch", True, torch.__version__)
    add("cuda", torch.cuda.is_available(), {
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    })
except Exception as exc:
    add("torch", False, str(exc))
for module in ("transformers", "peft", "accelerate", "bitsandbytes", "qwen_vl_utils"):
    add(module, importlib.util.find_spec(module) is not None, module)
add("model_path", model_path.exists(), model_path.as_posix())
if model_path.exists():
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path.as_posix(), trust_remote_code=True)
        add("qwen25vl_processor", True, processor.__class__.__name__)
    except Exception as exc:
        add("qwen25vl_processor", False, str(exc))
adapter = Path(r3)
add("r3_policy_init_adapter", (adapter / "adapter_config.json").exists(), adapter.as_posix())
add("preference_v8_1_data", preference.exists(), preference.as_posix())
if preference.exists():
    add("preference_v8_1_nonempty", True, sum(1 for line in preference.open(encoding="utf-8") if line.strip()))
if audit_path.exists():
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    add("preference_v8_1_train_ready", audit.get("train_ready") is True, audit.get("train_ready"))
    add("uses_model_predictions", audit.get("uses_model_predictions") is True, audit.get("uses_model_predictions"))
    add("actual_prediction_rate", float(audit.get("rejected_from_r3_actual_prediction_rate", 0)) > 0.90, audit.get("rejected_from_r3_actual_prediction_rate"))
    add("heldout_leakage_count", audit.get("heldout_leakage_count") == 0, audit.get("heldout_leakage_count"))
else:
    add("preference_v8_1_audit", False, f"missing {audit_path}")
add("dpo_v8_1_config", config_path.exists(), config_path.as_posix())
cfg = yaml_values(config_path)
if cfg:
    add("config_init_adapter", cfg.get("init_adapter") == r3, cfg.get("init_adapter"))
    add("config_reference_adapter", cfg.get("reference_adapter") == r3, cfg.get("reference_adapter"))
    add("config_reference_free", cfg.get("reference_free", "").lower() == "false", cfg.get("reference_free"))
    add("config_preference_file", cfg.get("preference_file") == preference.as_posix(), cfg.get("preference_file"))
add("heldout_eval_ids", heldout_path.exists(), heldout_path.as_posix())
if heldout_path.exists():
    payload = json.loads(heldout_path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    valid_ids = isinstance(ids, list) and len(ids) == 100 and len(ids) == len(set(ids))
    add("heldout_eval_id_count", valid_ids, len(ids) if isinstance(ids, list) else None)
    if valid_ids:
        for setting in ("normal", "text_only", "wrong_image", "blank_image"):
            path = heldout_pool / f"lingoqa_strict_{setting}.jsonl"
            found = set()
            if path.exists():
                found = {str(json.loads(line).get("id", "")) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
            add(f"heldout_pool_{setting}", path.exists() and set(ids).issubset(found), {"path": path.as_posix(), "count": len(found)})
result = {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}
Path("outputs/gpu_stage/stage8_dpo_v8_1_ready.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    sys.exit(1)
PY
