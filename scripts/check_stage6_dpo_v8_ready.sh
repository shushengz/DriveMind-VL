#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
CONFIG_PATH="${CONFIG_PATH:-configs/dpo_v8_lingo_smoke.yaml}"
mkdir -p outputs/gpu_stage

STAGE6_MODEL_PATH="$MODEL_PATH" STAGE6_CONFIG_PATH="$CONFIG_PATH" "$PYTHON_BIN" - <<'PY' 2>&1 | tee outputs/gpu_stage/stage6_dpo_v8_ready.log
import importlib.util
import json
import os
import sys
from pathlib import Path

model_path = Path(os.environ["STAGE6_MODEL_PATH"])
config_path = Path(os.environ["STAGE6_CONFIG_PATH"])
init_expected = "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
preference_path = Path("data/train/preference_v8/preference_v8_pairs.jsonl")
audit_path = Path("outputs/data_audit/preference_v8_audit.json")
heldout_path = Path("outputs/final_report/stage4_5_heldout_ids_100.json")
heldout_pool = Path("data/processed/visual_control_heldout")
errors, warnings, checks = [], [], {}

def add(name, ok, detail=None):
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        errors.append(f"{name}: {detail}")

def config_values(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                out[key.strip()] = value.strip()
    return out

add("python_executable", True, sys.executable)
try:
    import torch
    add("torch", True, torch.__version__)
    add("cuda", torch.cuda.is_available(), {
        "available": torch.cuda.is_available(),
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
adapter_path = Path(init_expected)
add("r3_policy_init_adapter", (adapter_path / "adapter_config.json").exists(), adapter_path.as_posix())
add("preference_v8_data", preference_path.exists(), preference_path.as_posix())
if preference_path.exists():
    pair_count = sum(1 for line in preference_path.open("r", encoding="utf-8") if line.strip())
    add("preference_v8_nonempty", pair_count > 0, {"pair_count": pair_count})
if audit_path.exists():
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    add("preference_v8_train_ready", audit.get("train_ready") is True, audit.get("train_ready"))
    add("heldout_leakage_count", audit.get("heldout_leakage_count") == 0, audit.get("heldout_leakage_count"))
else:
    add("preference_v8_audit", False, f"missing {audit_path}")
add("dpo_config", config_path.exists(), config_path.as_posix())
config = config_values(config_path)
if config:
    add("config_init_adapter", config.get("init_adapter") == init_expected, config.get("init_adapter"))
    add("config_reference_adapter", config.get("reference_adapter") == init_expected, config.get("reference_adapter"))
    add("config_reference_free", config.get("reference_free", "").lower() == "false", config.get("reference_free"))
add("heldout_eval_ids", heldout_path.exists(), heldout_path.as_posix())
if heldout_path.exists():
    payload = json.loads(heldout_path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    add("heldout_eval_id_count", isinstance(ids, list) and len(ids) == 100 and len(ids) == len(set(ids)), len(ids) if isinstance(ids, list) else None)
    for setting in ("normal", "text_only", "wrong_image", "blank_image"):
        path = heldout_pool / f"lingoqa_strict_{setting}.jsonl"
        present = set()
        if path.exists():
            for line in path.open("r", encoding="utf-8"):
                if line.strip():
                    present.add(str(json.loads(line).get("id", "")))
        add(f"heldout_pool_{setting}", path.exists() and set(ids).issubset(present), {"path": path.as_posix(), "count": len(present)})
result = {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}
Path("outputs/gpu_stage/stage6_dpo_v8_ready.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    sys.exit(1)
PY
