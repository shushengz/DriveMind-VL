#!/usr/bin/env bash
set -euo pipefail

echo "Stage 4.5 leakage audit and held-out eval only. No training, no DPO, no GRPO."

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
EVAL_SAMPLES=100
BF16=0
RUN=0
DO_AUDIT=0
DO_IDS=0
DO_EVAL=0
DO_MAIN=0
DO_DIAGNOSE=0
DO_GALLERY=0
SFT2_ADAPTER="checkpoints/qwen25vl_lora_sft_v2/"
DPO_ADAPTER="outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v7_grounded/"
R2_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r2_lingo_smoke/"
R3_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --audit_overlap) DO_AUDIT=1; shift ;;
    --build_ids) DO_IDS=1; shift ;;
    --eval) DO_EVAL=1; shift ;;
    --main_results) DO_MAIN=1; shift ;;
    --diagnose) DO_DIAGNOSE=1; shift ;;
    --case_gallery) DO_GALLERY=1; shift ;;
    --all) DO_AUDIT=1; DO_IDS=1; DO_EVAL=1; DO_MAIN=1; DO_DIAGNOSE=1; DO_GALLERY=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    --eval_samples) EVAL_SAMPLES="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --sft2_adapter) SFT2_ADAPTER="$2"; shift 2 ;;
    --dpo_adapter) DPO_ADAPTER="$2"; shift 2 ;;
    --r2_adapter) R2_ADAPTER="$2"; shift 2 ;;
    --r3_adapter) R3_ADAPTER="$2"; shift 2 ;;
    *) echo "[stage4.5] unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DO_AUDIT$DO_IDS$DO_EVAL$DO_MAIN$DO_DIAGNOSE$DO_GALLERY" == "000000" ]]; then
  DO_AUDIT=1; DO_IDS=1; DO_EVAL=1; DO_MAIN=1; DO_DIAGNOSE=1; DO_GALLERY=1
fi
if [[ "$EVAL_SAMPLES" != "100" && "$EVAL_SAMPLES" != "200" ]]; then
  echo "[stage4.5] --eval_samples must be 100 or 200" >&2
  exit 2
fi

MODE="dry_run"; [[ "$RUN" == "1" ]] && MODE="run"
echo "[stage4.5] mode=${MODE} eval_samples=${EVAL_SAMPLES} model_path=${MODEL_PATH}"
if [[ "$RUN" == "1" ]]; then
  REPORT_ROOT="outputs/final_report"
  VISUAL_CONTROL_DIR="data/processed/visual_control_heldout"
  PREDICTION_ROOT="outputs/predictions_heldout"
  EVAL_OUTPUT_ROOT="outputs/stage4_5_intermediate"
else
  REPORT_ROOT="outputs/stage4_5_dry_run"
  VISUAL_CONTROL_DIR="${REPORT_ROOT}/visual_control_heldout"
  PREDICTION_ROOT="${REPORT_ROOT}/predictions_heldout"
  EVAL_OUTPUT_ROOT="${REPORT_ROOT}/intermediate"
fi
mkdir -p "$REPORT_ROOT" "$PREDICTION_ROOT" "$EVAL_OUTPUT_ROOT"

if [[ "$DO_AUDIT" == "1" ]]; then
  echo "[stage4.5] 1/7 audit train/eval overlap"
  "$PYTHON_BIN" src/eval/audit_train_eval_overlap.py \
    --output_json "${REPORT_ROOT}/stage4_5_train_eval_overlap.json" \
    --output_csv "${REPORT_ROOT}/stage4_5_train_eval_overlap.csv" \
    --output_md "${REPORT_ROOT}/stage4_5_train_eval_overlap.md"
fi

if [[ "$DO_IDS" == "1" ]]; then
  echo "[stage4.5] 2/7 build untouched strict held-out pool and IDs"
  "$PYTHON_BIN" src/eval/build_stage4_5_heldout_pool.py --output_dir "$VISUAL_CONTROL_DIR"
  "$PYTHON_BIN" src/eval/build_heldout_eval_ids.py --visual_control_dir "$VISUAL_CONTROL_DIR" --output_dir "$REPORT_ROOT"
fi

IDS_FILE="${REPORT_ROOT}/stage4_5_heldout_ids_${EVAL_SAMPLES}.json"
SELECTED="$("$PYTHON_BIN" - "$IDS_FILE" <<'PY'
import json, sys
from pathlib import Path
path=Path(sys.argv[1])
if not path.exists():
    print(-1)
else:
    print(int(json.loads(path.read_text(encoding="utf-8")).get("selected", 0)))
PY
)"
EVAL_BLOCKED=0
if [[ "$DO_EVAL" == "1" && "$SELECTED" -lt "$EVAL_SAMPLES" ]]; then
  EVAL_BLOCKED=1
  "$PYTHON_BIN" - "$IDS_FILE" "$EVAL_SAMPLES" "$REPORT_ROOT" <<'PY'
import json, sys
from pathlib import Path
ids_path=Path(sys.argv[1]); required=int(sys.argv[2]); report_root=Path(sys.argv[3])
payload=json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.exists() else {}
out={
  "blocked": True,
  "reason": "insufficient non-overlapping held-out IDs; no model inference was run",
  "required": required,
  "selected": int(payload.get("selected", 0)),
  "warnings": payload.get("warnings", []),
}
(report_root / "stage4_5_heldout_eval_blocked.json").write_text(json.dumps(out, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
  echo "[stage4.5] held-out eval blocked: need ${EVAL_SAMPLES} non-overlapping IDs, found ${SELECTED}."
fi

eval_one() {
  local name="$1"
  local adapter="$2"
  local args=(
    src/eval/run_visual_control_eval.py
    --dataset lingoqa
    --visual_control_dir "$VISUAL_CONTROL_DIR"
    --eval_ids "$IDS_FILE"
    --model_path "$MODEL_PATH"
    --model_name "$name"
    --prediction_root "$PREDICTION_ROOT"
    --output_root "$EVAL_OUTPUT_ROOT"
  )
  if [[ -n "$adapter" ]]; then
    if [[ ! -f "${adapter%/}/adapter_config.json" ]]; then
      echo "[stage4.5] adapter missing for ${name}: ${adapter}" >&2
      return 1
    fi
    args+=(--adapter_path "$adapter")
  fi
  [[ "$BF16" == "1" ]] && args+=(--bf16)
  [[ "$RUN" != "1" ]] && args+=(--dry_run)
  "$PYTHON_BIN" "${args[@]}"
}

if [[ "$DO_EVAL" == "1" && "$EVAL_BLOCKED" == "0" ]]; then
  echo "[stage4.5] 3/7 run held-out strict visual-control eval"
  eval_one base_qwen25vl_3b ""
  eval_one sft_v2 "$SFT2_ADAPTER"
  eval_one dpo_v7 "$DPO_ADAPTER"
  eval_one sft_v3_r2_lingo_smoke "$R2_ADAPTER"
  eval_one sft_v3_r3_lingo_smoke "$R3_ADAPTER"
fi

if [[ "$DO_MAIN" == "1" ]]; then
  echo "[stage4.5] 4/7 build held-out main results"
  MAIN_ARGS=(src/eval/build_stage4_5_heldout_results.py --prediction_root "$PREDICTION_ROOT" --output "${REPORT_ROOT}/stage4_5_heldout_main_results.csv" --warnings_output "${REPORT_ROOT}/stage4_5_heldout_main_results_warnings.json")
  [[ "$RUN" != "1" ]] && MAIN_ARGS+=(--dry_run)
  "$PYTHON_BIN" "${MAIN_ARGS[@]}"
fi

if [[ "$DO_DIAGNOSE" == "1" ]]; then
  echo "[stage4.5] 5/7 diagnose held-out result"
  "$PYTHON_BIN" src/eval/diagnose_stage4_5_heldout.py \
    --overlap "${REPORT_ROOT}/stage4_5_train_eval_overlap.json" \
    --results "${REPORT_ROOT}/stage4_5_heldout_main_results.csv" \
    --output_json "${REPORT_ROOT}/stage4_5_heldout_diagnosis.json" \
    --output_md "${REPORT_ROOT}/stage4_5_heldout_diagnosis.md"
fi

if [[ "$DO_GALLERY" == "1" ]]; then
  echo "[stage4.5] 6/7 build held-out case gallery"
  "$PYTHON_BIN" src/eval/build_stage4_5_heldout_case_gallery.py \
    --prediction_root "$PREDICTION_ROOT" \
    --output_csv "${REPORT_ROOT}/stage4_5_heldout_case_gallery.csv" \
    --output_html "${REPORT_ROOT}/stage4_5_heldout_case_gallery.html"
fi

echo "[stage4.5] done; no DPO or GRPO action was taken."
