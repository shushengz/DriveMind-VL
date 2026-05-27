#!/usr/bin/env bash
set -euo pipefail

echo "GPU Stage 10 smoke: case-gap-aware DPO-v8.2 only. Exactly 25 steps; no GRPO and no long training."
export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
INIT_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
REFERENCE_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
PREFERENCE="data/train/preference_v8_2/preference_v8_2_pairs.jsonl"
EVAL_IDS="outputs/final_report/stage4_5_heldout_ids_100.json"
MAX_STEPS=25
RUN=0; BF16=0; QLORA=0
DO_CHECK=0; DO_TRAIN=0; DO_EVAL=0; DO_RESULTS=0; DO_DIAGNOSE=0; DO_GALLERY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1; shift ;;
    --train) DO_TRAIN=1; shift ;;
    --eval) DO_EVAL=1; shift ;;
    --results) DO_RESULTS=1; shift ;;
    --diagnose) DO_DIAGNOSE=1; shift ;;
    --case_gallery) DO_GALLERY=1; shift ;;
    --all) DO_CHECK=1; DO_TRAIN=1; DO_EVAL=1; DO_RESULTS=1; DO_DIAGNOSE=1; DO_GALLERY=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --eval_ids) EVAL_IDS="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --qlora) QLORA=1; shift ;;
    *) echo "[stage10] unknown or forbidden argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_CHECK$DO_TRAIN$DO_EVAL$DO_RESULTS$DO_DIAGNOSE$DO_GALLERY" == "000000" ]]; then
  DO_CHECK=1; DO_TRAIN=1; DO_EVAL=1; DO_RESULTS=1; DO_DIAGNOSE=1; DO_GALLERY=1
fi
MODE="dry_run"; [[ "$RUN" == "1" ]] && MODE="run"
echo "[stage10] mode=${MODE} fixed_max_steps=${MAX_STEPS} eval_ids=${EVAL_IDS}"

if [[ "$RUN" == "1" ]]; then
  PRED_ROOT="outputs/predictions_heldout"; EVAL_ROOT="outputs"
  TRAIN_OUTPUT="checkpoints/qwen25vl_lora_dpo_v8_2_lingo_smoke/"; TRAIN_LOGS="outputs/train_logs/dpo_v8_2_lingo_smoke/"
  RESULTS="outputs/final_report/stage10_dpo_v8_2_heldout_main_results.csv"
  WARNINGS="outputs/final_report/stage10_dpo_v8_2_heldout_main_results_warnings.json"
  DIAG_JSON="outputs/final_report/stage10_dpo_v8_2_diagnosis.json"; DIAG_MD="outputs/final_report/stage10_dpo_v8_2_diagnosis.md"
  GALLERY_CSV="outputs/final_report/stage10_dpo_v8_2_case_gallery.csv"; GALLERY_HTML="outputs/final_report/stage10_dpo_v8_2_case_gallery.html"
else
  PRED_ROOT="outputs/stage10_dry_run/predictions_heldout"; EVAL_ROOT="outputs/stage10_dry_run"
  TRAIN_OUTPUT="outputs/stage10_dry_run/checkpoints/qwen25vl_lora_dpo_v8_2_lingo_smoke/"; TRAIN_LOGS="outputs/stage10_dry_run/train_logs/dpo_v8_2_lingo_smoke/"
  RESULTS="outputs/stage10_dry_run/stage10_dpo_v8_2_heldout_main_results.csv"
  WARNINGS="outputs/stage10_dry_run/stage10_dpo_v8_2_heldout_main_results_warnings.json"
  DIAG_JSON="outputs/stage10_dry_run/stage10_dpo_v8_2_diagnosis.json"; DIAG_MD="outputs/stage10_dry_run/stage10_dpo_v8_2_diagnosis.md"
  GALLERY_CSV="outputs/stage10_dry_run/stage10_dpo_v8_2_case_gallery.csv"; GALLERY_HTML="outputs/stage10_dry_run/stage10_dpo_v8_2_case_gallery.html"
fi

echo "[stage10] 1/6 readiness gate"
MODEL_PATH="$MODEL_PATH" CONFIG_PATH="configs/dpo_v8_2_lingo_smoke.yaml" PYTHON_BIN="$PYTHON_BIN" bash scripts/check_stage10_dpo_v8_2_ready.sh

if [[ "$DO_TRAIN" == "1" ]]; then
  echo "[stage10] 2/6 DPO-v8.2 train (fixed 25 steps)"
  TRAIN_ARGS=(src/train/train_qwen25vl_dpo_v8_lora.py --config configs/dpo_v8_2_lingo_smoke.yaml --model_path "$MODEL_PATH" --init_adapter "$INIT_ADAPTER" --reference_adapter "$REFERENCE_ADAPTER" --preference_file "$PREFERENCE" --output_dir "$TRAIN_OUTPUT" --log_dir "$TRAIN_LOGS" --max_steps "$MAX_STEPS" --save_steps "$MAX_STEPS")
  [[ "$BF16" == "1" ]] && TRAIN_ARGS+=(--bf16)
  [[ "$QLORA" == "1" ]] && TRAIN_ARGS+=(--qlora)
  [[ "$RUN" != "1" ]] && TRAIN_ARGS+=(--dry_run)
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" - "$TRAIN_LOGS/train_summary.json" <<'PY'
import json
import sys
from pathlib import Path
summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not summary.get("success") or summary.get("nan_detected") or summary.get("oom") or summary.get("max_steps") != 25:
    raise SystemExit(f"Stage 10 training failed safety gate: {summary}")
print("[stage10] training summary safety gate passed")
PY
  fi
fi

if [[ "$DO_EVAL" == "1" ]]; then
  echo "[stage10] 3/6 held-out strict visual-control eval"
  TARGET="$PRED_ROOT/lingoqa/dpo_v8_2_lingo_smoke_step25/strict_visual"
  if [[ "$RUN" == "1" ]] && [[ -e "$TARGET/normal.jsonl" || -e "$TARGET/text_only.jsonl" || -e "$TARGET/wrong_image.jsonl" || -e "$TARGET/blank_image.jsonl" ]]; then
    echo "[stage10] refusing to overwrite existing DPO-v8.2 predictions: $TARGET" >&2
    exit 1
  fi
  [[ "$RUN" != "1" ]] || [[ -e "$TRAIN_OUTPUT/adapter_config.json" ]] || { echo "[stage10] missing trained adapter: $TRAIN_OUTPUT" >&2; exit 1; }
  EVAL_ARGS=(src/eval/run_visual_control_eval.py --dataset lingoqa --visual_control_dir data/processed/visual_control_heldout --eval_ids "$EVAL_IDS" --model_path "$MODEL_PATH" --adapter_path "$TRAIN_OUTPUT" --model_name dpo_v8_2_lingo_smoke_step25 --prediction_root "$PRED_ROOT" --output_root "$EVAL_ROOT" --mode strict_visual --max_new_tokens 64)
  [[ "$BF16" == "1" ]] && EVAL_ARGS+=(--bf16)
  [[ "$RUN" != "1" ]] && EVAL_ARGS+=(--dry_run)
  "$PYTHON_BIN" "${EVAL_ARGS[@]}"
fi
if [[ "$DO_RESULTS" == "1" ]]; then
  echo "[stage10] 4/6 build main results"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" src/eval/build_stage10_dpo_v8_2_results.py --prediction_root "$PRED_ROOT" --output "$RESULTS" --warnings_output "$WARNINGS"
  else
    echo "[stage10] dry-run: results deferred until real predictions exist."
  fi
fi
if [[ "$DO_DIAGNOSE" == "1" ]]; then
  echo "[stage10] 5/6 diagnosis"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" src/eval/diagnose_stage10_dpo_v8_2.py --results "$RESULTS" --train_summary "$TRAIN_LOGS/train_summary.json" --output_json "$DIAG_JSON" --output_md "$DIAG_MD"
  else
    echo "[stage10] dry-run: diagnosis deferred."
  fi
fi
if [[ "$DO_GALLERY" == "1" ]]; then
  echo "[stage10] 6/6 case gallery"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" src/eval/build_stage10_dpo_v8_2_case_gallery.py --prediction_root "$PRED_ROOT" --output_csv "$GALLERY_CSV" --output_html "$GALLERY_HTML"
  else
    echo "[stage10] dry-run: gallery deferred."
  fi
fi
echo "[stage10] complete. No GRPO-lite command exists in this workflow."
