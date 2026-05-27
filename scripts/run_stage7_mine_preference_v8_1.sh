#!/usr/bin/env bash
set -euo pipefail

echo "Stage 7 mining: r3 prediction only when --predict --run is selected. No DPO/SFT/GRPO training."
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
MAX_MINING_IDS=300
SEED=42
RUN=0
BF16=0
DO_POOL=0
DO_PREDICT=0
DO_MINE=0
DO_BUILD=0
DO_AUDIT=0
DO_CONFIG=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build_pool) DO_POOL=1; shift ;;
    --predict) DO_PREDICT=1; shift ;;
    --mine) DO_MINE=1; shift ;;
    --build_pairs) DO_BUILD=1; shift ;;
    --audit) DO_AUDIT=1; shift ;;
    --make_config) DO_CONFIG=1; shift ;;
    --all) DO_POOL=1; DO_PREDICT=1; DO_MINE=1; DO_BUILD=1; DO_AUDIT=1; DO_CONFIG=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --max_mining_ids) MAX_MINING_IDS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    *) echo "[stage7] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DO_POOL$DO_PREDICT$DO_MINE$DO_BUILD$DO_AUDIT$DO_CONFIG" == "000000" ]]; then
  DO_POOL=1; DO_PREDICT=1; DO_MINE=1; DO_BUILD=1; DO_AUDIT=1; DO_CONFIG=1
fi

if [[ "$RUN" == "1" ]]; then
  MINING_DIR="data/mining/v8_1"
  PREFERENCE_DIR="data/train/preference_v8_1"
  PREDICTION_ROOT="outputs/predictions_mining"
  AUDIT_DIR="outputs/data_audit"
  EVAL_OUTPUT_ROOT="outputs/mining_eval"
else
  MINING_DIR="outputs/stage7_dry_run/data/mining/v8_1"
  PREFERENCE_DIR="outputs/stage7_dry_run/data/train/preference_v8_1"
  PREDICTION_ROOT="outputs/stage7_dry_run/predictions_mining"
  AUDIT_DIR="outputs/stage7_dry_run/data_audit"
  EVAL_OUTPUT_ROOT="outputs/stage7_dry_run"
fi
IDS_FILE="$MINING_DIR/lingoqa_mining_pool_ids.json"
HARD_FILE="$MINING_DIR/hard_negatives_v8_1.jsonl"
FORMAL_PREDICTIONS="outputs/predictions_mining/lingoqa/sft_v3_r3_lingo_smoke/strict_visual/normal.jsonl"
mkdir -p "$MINING_DIR" "$PREFERENCE_DIR" "$AUDIT_DIR"

if [[ "$DO_POOL" == "1" ]]; then
  echo "[stage7] 1/6 build non-heldout mining pool"
  POOL_ARGS=(
    src/data/build_mining_pool_v8_1.py
    --output_ids "$IDS_FILE"
    --output_summary "$MINING_DIR/lingoqa_mining_pool_summary.json"
    --max_ids "$MAX_MINING_IDS"
    --seed "$SEED"
  )
  if [[ "$RUN" != "1" ]]; then POOL_ARGS+=(--dry_run); fi
  "$PYTHON_BIN" "${POOL_ARGS[@]}"
fi

if [[ "$DO_PREDICT" == "1" ]]; then
  echo "[stage7] 2/6 r3 mining predictions"
  EVAL_ARGS=(
    src/eval/run_visual_control_eval.py
    --dataset lingoqa
    --eval_ids "$IDS_FILE"
    --visual_control_dir data/processed/visual_control
    --model_path "$MODEL_PATH"
    --adapter_path checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/
    --model_name sft_v3_r3_lingo_smoke
    --prediction_root "$PREDICTION_ROOT"
    --output_root "$EVAL_OUTPUT_ROOT"
    --mode strict_visual
  )
  if [[ "$BF16" == "1" ]]; then EVAL_ARGS+=(--bf16); fi
  if [[ "$RUN" != "1" ]]; then EVAL_ARGS+=(--dry_run); fi
  "$PYTHON_BIN" "${EVAL_ARGS[@]}"
fi

can_mine=0
if [[ "$RUN" == "1" && -f "$FORMAL_PREDICTIONS" ]]; then
  can_mine=1
fi

if [[ "$DO_MINE" == "1" ]]; then
  echo "[stage7] 3/6 mine answer-only hard negatives"
  if [[ "$can_mine" == "1" ]]; then
    "$PYTHON_BIN" src/data/mine_preference_hard_negatives_v8_1.py \
      --mining_ids "$IDS_FILE" \
      --output "$HARD_FILE" \
      --stats "$MINING_DIR/hard_negatives_v8_1_stats.json" \
      --report "$AUDIT_DIR/hard_negatives_v8_1_report.md"
  else
    echo "[stage7] pending: formal r3 mining predictions are required before hard-negative mining."
  fi
fi

if [[ "$DO_BUILD" == "1" ]]; then
  echo "[stage7] 4/6 build Preference-v8.1 pairs"
  if [[ "$can_mine" == "1" && -f "$HARD_FILE" ]]; then
    "$PYTHON_BIN" src/data/build_preference_v8_1.py \
      --hard_negatives "$HARD_FILE" \
      --output_dir "$PREFERENCE_DIR" \
      --seed "$SEED"
  else
    echo "[stage7] pending: model-mined hard negatives are required before pair construction."
  fi
fi

if [[ "$DO_AUDIT" == "1" ]]; then
  echo "[stage7] 5/6 audit Preference-v8.1"
  if [[ "$can_mine" == "1" && -f "$PREFERENCE_DIR/preference_v8_1_pairs.jsonl" ]]; then
    "$PYTHON_BIN" src/data/audit_preference_v8_1.py \
      --input "$PREFERENCE_DIR/preference_v8_1_pairs.jsonl" \
      --hard_negatives "$HARD_FILE" \
      --output_json "$AUDIT_DIR/preference_v8_1_audit.json" \
      --output_md "$AUDIT_DIR/preference_v8_1_audit.md"
  else
    echo "[stage7] pending: Preference-v8.1 pairs are not available for audit."
  fi
fi

if [[ "$DO_CONFIG" == "1" ]]; then
  echo "[stage7] 6/6 verify DPO-v8.1 configuration draft (no training)"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
path = Path("configs/dpo_v8_1_lingo_smoke.yaml")
text = path.read_text(encoding="utf-8") if path.exists() else ""
required = [
    "init_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
    "reference_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
    "preference_file: data/train/preference_v8_1/preference_v8_1_pairs.jsonl",
    "reference_free: false",
]
missing = [entry for entry in required if entry not in text]
if missing:
    raise SystemExit("invalid DPO-v8.1 config draft: missing " + ", ".join(missing))
print("[stage7] DPO-v8.1 draft uses frozen r3 reference; training is not launched.")
PY
fi

echo "[stage7] complete. No DPO/SFT/GRPO training has been launched."
