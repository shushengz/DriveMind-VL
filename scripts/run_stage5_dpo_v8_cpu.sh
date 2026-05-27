#!/usr/bin/env bash
set -euo pipefail

echo "CPU-only Stage 5: no model training, no model inference, no GPU usage."

PYTHON_BIN="${PYTHON_BIN:-python}"
MAX_PAIRS=500
SEED=42
RUN=false
DO_EXCLUDE=false
DO_BUILD=false
DO_AUDIT=false
DO_CONFIG=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exclude_heldout) DO_EXCLUDE=true ;;
    --build_pairs) DO_BUILD=true ;;
    --audit) DO_AUDIT=true ;;
    --make_config) DO_CONFIG=true ;;
    --all) DO_EXCLUDE=true; DO_BUILD=true; DO_AUDIT=true; DO_CONFIG=true ;;
    --run) RUN=true ;;
    --dry_run) RUN=false ;;
    --max_pairs) MAX_PAIRS="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    *) echo "[stage5] unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$DO_EXCLUDE" == false && "$DO_BUILD" == false && "$DO_AUDIT" == false && "$DO_CONFIG" == false ]]; then
  DO_EXCLUDE=true
  DO_BUILD=true
  DO_AUDIT=true
  DO_CONFIG=true
fi

HELDOUT_IDS="outputs/final_report/stage4_5_heldout_ids_100.json"
if [[ "$RUN" == true ]]; then
  echo "[stage5] mode=run max_pairs=${MAX_PAIRS} seed=${SEED}"
  PAIR_DIR="data/train/preference_v8"
  AUDIT_DIR="outputs/data_audit"
  DRY_ARGS=()
else
  echo "[stage5] mode=dry_run max_pairs=${MAX_PAIRS} seed=${SEED}"
  PAIR_DIR="outputs/stage5_dry_run/data/train/preference_v8"
  AUDIT_DIR="outputs/stage5_dry_run/data_audit"
  DRY_ARGS=(--dry_run)
fi

mkdir -p "$PAIR_DIR" "$AUDIT_DIR"

if [[ "$DO_EXCLUDE" == true ]]; then
  echo "[stage5] 1/4 exclude held-out IDs"
  "$PYTHON_BIN" src/data/exclude_heldout_ids.py \
    --candidate_file data/processed/visual_control/lingoqa_strict_normal.jsonl \
    --heldout_ids "$HELDOUT_IDS" \
    --filtered_output "$PAIR_DIR/eligible_normal_pool.jsonl" \
    --output_json "$AUDIT_DIR/dpo_v8_heldout_exclusion.json" \
    --output_md "$AUDIT_DIR/dpo_v8_heldout_exclusion.md" \
    "${DRY_ARGS[@]}"
fi

if [[ "$DO_BUILD" == true ]]; then
  echo "[stage5] 2/4 build answer-only Preference-v8 pairs"
  "$PYTHON_BIN" src/data/build_preference_v8.py \
    --visual_control_dir data/processed/visual_control \
    --heldout_ids "$HELDOUT_IDS" \
    --output_dir "$PAIR_DIR" \
    --max_pairs "$MAX_PAIRS" \
    --seed "$SEED" \
    "${DRY_ARGS[@]}"
fi

if [[ "$DO_AUDIT" == true ]]; then
  echo "[stage5] 3/4 audit Preference-v8 pairs"
  "$PYTHON_BIN" src/data/audit_preference_v8.py \
    --input "$PAIR_DIR/preference_v8_pairs.jsonl" \
    --heldout_ids "$HELDOUT_IDS" \
    --output_json "$AUDIT_DIR/preference_v8_audit.json" \
    --output_md "$AUDIT_DIR/preference_v8_audit.md" \
    "${DRY_ARGS[@]}"
fi

if [[ "$DO_CONFIG" == true ]]; then
  echo "[stage5] 4/4 verify DPO-v8 configuration draft (no training)"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path

path = Path("configs/dpo_v8_lingo_smoke.yaml")
if not path.exists():
    raise SystemExit(f"missing configuration draft: {path}")
text = path.read_text(encoding="utf-8")
required = (
    "init_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
    "reference_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
    "method: dpo",
    "reference_free: false",
)
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit("invalid DPO-v8 configuration draft: missing " + ", ".join(missing))
print("[stage5] policy and frozen reference are both configured from the r3 adapter.")
PY
fi

echo "[stage5] finished. This script has not loaded a model or launched training/inference."
