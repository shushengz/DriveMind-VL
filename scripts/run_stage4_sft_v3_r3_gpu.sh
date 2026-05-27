#!/usr/bin/env bash
set -euo pipefail

echo "GPU Stage 4 smoke: SFT-v3-r3 lightweight calibration only. No DPO, no GRPO, no large-scale training."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
INIT_ADAPTER="checkpoints/qwen25vl_lora_sft_v2/"
MAX_STEPS=100
EVAL_SAMPLES=100
RUN=0
DO_TRAIN=0
DO_EVAL=0
DO_MAIN=0
DO_DIAGNOSE=0
DO_GALLERY=0
BF16=0
QLORA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --train) DO_TRAIN=1; shift ;;
    --eval) DO_EVAL=1; shift ;;
    --main_results) DO_MAIN=1; shift ;;
    --diagnose) DO_DIAGNOSE=1; shift ;;
    --case_gallery) DO_GALLERY=1; shift ;;
    --all) DO_TRAIN=1; DO_EVAL=1; DO_MAIN=1; DO_DIAGNOSE=1; DO_GALLERY=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --init_adapter) INIT_ADAPTER="$2"; shift 2 ;;
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --eval_samples) EVAL_SAMPLES="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --qlora) QLORA=1; shift ;;
    *) echo "[stage4-r3] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DO_TRAIN$DO_EVAL$DO_MAIN$DO_DIAGNOSE$DO_GALLERY" == "00000" ]]; then
  DO_TRAIN=1; DO_EVAL=1; DO_MAIN=1; DO_DIAGNOSE=1; DO_GALLERY=1
fi

MODE="dry_run"
if [[ "$RUN" == "1" ]]; then MODE="run"; fi
echo "[stage4-r3] mode=${MODE} model_path=${MODEL_PATH} init_adapter=${INIT_ADAPTER} max_steps=${MAX_STEPS} eval_samples=${EVAL_SAMPLES}"

mkdir -p outputs/gpu_stage outputs/train_logs/sft_v3_r3_lingo_smoke

check_ready() {
  STAGE4_MODEL_PATH="$MODEL_PATH" STAGE4_INIT_ADAPTER="$INIT_ADAPTER" "$PYTHON_BIN" - <<'PY'
import importlib.util
import json
import os
import sys
from pathlib import Path

model_path = Path(os.environ["STAGE4_MODEL_PATH"])
init_adapter = Path(os.environ["STAGE4_INIT_ADAPTER"])
audit_path = Path("outputs/data_audit/sft_v3_r3_lingo_audit.json")
train_file = Path("data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
train_script = Path("src/train/train_qwen25vl_sft_v3_lora.py")

errors = []
warnings = []
checks = {}

def add(name, ok, detail=None):
    checks[name] = {"ok": bool(ok)}
    if detail is not None:
        checks[name]["detail"] = detail
    if not ok:
        errors.append(f"{name}: {detail or 'failed'}")

add("python_executable", True, sys.executable)
try:
    import torch
    add("torch", True, getattr(torch, "__version__", "unknown"))
    add("cuda", torch.cuda.is_available(), {"available": torch.cuda.is_available(), "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0})
except Exception as exc:
    add("torch", False, str(exc))

for mod in ["transformers", "peft", "accelerate", "bitsandbytes", "qwen_vl_utils"]:
    add(mod, importlib.util.find_spec(mod) is not None)

add("model_path", model_path.exists(), model_path.as_posix())
if model_path.exists():
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path.as_posix(), trust_remote_code=True)
        add("qwen25vl_processor", True, proc.__class__.__name__)
    except Exception as exc:
        add("qwen25vl_processor", False, str(exc))

add("r3_train_file", train_file.exists(), train_file.as_posix())
if audit_path.exists():
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    train_ready = bool(audit.get("train_ready"))
    add("r3_audit_train_ready", train_ready, {"train_ready": train_ready, "path": audit_path.as_posix()})
else:
    add("r3_audit_train_ready", False, f"missing {audit_path.as_posix()}")

add("init_adapter", (init_adapter / "adapter_config.json").exists(), init_adapter.as_posix())
if train_script.exists():
    text = train_script.read_text(encoding="utf-8")
    add("train_script_init_adapter_support", "--init_adapter" in text and "PeftModel.from_pretrained" in text, train_script.as_posix())
else:
    add("train_script_init_adapter_support", False, train_script.as_posix())

result = {"ok": not errors, "errors": errors, "warnings": warnings, "checks": checks}
Path("outputs/gpu_stage/stage4_r3_gpu_ready.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
if errors:
    sys.exit(1)
PY
}

echo "[stage4-r3] 1/6 readiness check"
check_ready 2>&1 | tee outputs/gpu_stage/stage4_r3_gpu_ready.log

if [[ "$DO_TRAIN" == "1" ]]; then
  echo "[stage4-r3] 2/6 train r3 from init adapter"
  TRAIN_ARGS=(
    src/train/train_qwen25vl_sft_v3_lora.py
    --config configs/sft_v3_r3_lingo_smoke.yaml
    --model_path "$MODEL_PATH"
    --train_file data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl
    --init_adapter "$INIT_ADAPTER"
    --output_dir checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/
    --log_dir outputs/train_logs/sft_v3_r3_lingo_smoke/
    --max_steps "$MAX_STEPS"
  )
  if [[ "$QLORA" == "1" ]]; then TRAIN_ARGS+=(--qlora); fi
  if [[ "$BF16" == "1" ]]; then TRAIN_ARGS+=(--bf16); fi
  if [[ "$RUN" != "1" ]]; then TRAIN_ARGS+=(--dry_run); fi
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
fi

if [[ "$DO_EVAL" == "1" ]]; then
  echo "[stage4-r3] 3/6 strict visual-control eval"
  EVAL_ARGS=(
    src/eval/run_visual_control_eval.py
    --dataset lingoqa
    --eval_samples "$EVAL_SAMPLES"
    --model_path "$MODEL_PATH"
    --adapter_path checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/
    --model_name sft_v3_r3_lingo_smoke
    --output_root outputs
  )
  if [[ "$BF16" == "1" ]]; then EVAL_ARGS+=(--bf16); fi
  if [[ "$RUN" != "1" ]]; then EVAL_ARGS+=(--dry_run); fi
  "$PYTHON_BIN" "${EVAL_ARGS[@]}"
fi

if [[ "$DO_MAIN" == "1" ]]; then
  echo "[stage4-r3] 4/6 build main results"
  "$PYTHON_BIN" src/eval/build_stage4_main_results.py --dataset lingoqa
fi

if [[ "$DO_DIAGNOSE" == "1" ]]; then
  echo "[stage4-r3] 5/6 diagnose"
  "$PYTHON_BIN" src/eval/diagnose_stage4_sft_v3_r3.py
fi

if [[ "$DO_GALLERY" == "1" ]]; then
  echo "[stage4-r3] 6/6 case gallery"
  "$PYTHON_BIN" src/eval/build_stage4_sft_v3_r3_case_gallery.py
fi

echo "[stage4-r3] done"
