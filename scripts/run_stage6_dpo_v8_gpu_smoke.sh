#!/usr/bin/env bash
set -euo pipefail

echo "GPU Stage 6 smoke: DPO-v8 only. No GRPO, no large-scale training, and held-out eval only."
export OMP_NUM_THREADS=1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
INIT_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
REFERENCE_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
EVAL_IDS="outputs/final_report/stage4_5_heldout_ids_100.json"
MAX_STEPS=50
RUN=0
BF16=0
QLORA=0
DO_CHECK=0
DO_TRAIN=0
DO_EVAL=0
DO_RESULTS=0
DO_DIAGNOSE=0
DO_GALLERY=0

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
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --eval_ids) EVAL_IDS="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --qlora) QLORA=1; shift ;;
    *) echo "[stage6] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DO_CHECK$DO_TRAIN$DO_EVAL$DO_RESULTS$DO_DIAGNOSE$DO_GALLERY" == "000000" ]]; then
  DO_CHECK=1; DO_TRAIN=1; DO_EVAL=1; DO_RESULTS=1; DO_DIAGNOSE=1; DO_GALLERY=1
fi

MODE="dry_run"
if [[ "$RUN" == "1" ]]; then MODE="run"; fi
echo "[stage6] mode=${MODE} max_steps=${MAX_STEPS} eval_ids=${EVAL_IDS}"

if [[ "$RUN" == "1" ]]; then
  PREDICTION_ROOT="outputs/predictions_heldout"
  EVAL_OUTPUT_ROOT="outputs"
  TRAIN_OUTPUT="checkpoints/qwen25vl_lora_dpo_v8_lingo_smoke/"
  TRAIN_LOGS="outputs/train_logs/dpo_v8_lingo_smoke/"
  RESULTS_OUTPUT="outputs/final_report/stage6_dpo_v8_heldout_main_results.csv"
  WARNINGS_OUTPUT="outputs/final_report/stage6_dpo_v8_heldout_main_results_warnings.json"
  DIAGNOSIS_JSON="outputs/final_report/stage6_dpo_v8_diagnosis.json"
  DIAGNOSIS_MD="outputs/final_report/stage6_dpo_v8_diagnosis.md"
  GALLERY_CSV="outputs/final_report/stage6_dpo_v8_case_gallery.csv"
  GALLERY_HTML="outputs/final_report/stage6_dpo_v8_case_gallery.html"
else
  PREDICTION_ROOT="outputs/stage6_dry_run/predictions_heldout"
  EVAL_OUTPUT_ROOT="outputs/stage6_dry_run"
  TRAIN_OUTPUT="outputs/stage6_dry_run/checkpoints/qwen25vl_lora_dpo_v8_lingo_smoke/"
  TRAIN_LOGS="outputs/stage6_dry_run/train_logs/dpo_v8_lingo_smoke/"
  RESULTS_OUTPUT="outputs/stage6_dry_run/stage6_dpo_v8_heldout_main_results.csv"
  WARNINGS_OUTPUT="outputs/stage6_dry_run/stage6_dpo_v8_heldout_main_results_warnings.json"
  DIAGNOSIS_JSON="outputs/stage6_dry_run/stage6_dpo_v8_diagnosis.json"
  DIAGNOSIS_MD="outputs/stage6_dry_run/stage6_dpo_v8_diagnosis.md"
  GALLERY_CSV="outputs/stage6_dry_run/stage6_dpo_v8_case_gallery.csv"
  GALLERY_HTML="outputs/stage6_dry_run/stage6_dpo_v8_case_gallery.html"
fi

echo "[stage6] 1/6 readiness"
MODEL_PATH="$MODEL_PATH" PYTHON_BIN="$PYTHON_BIN" bash scripts/check_stage6_dpo_v8_ready.sh

if [[ "$DO_TRAIN" == "1" ]]; then
  echo "[stage6] 2/6 DPO-v8 smoke train"
  TRAIN_ARGS=(
    src/train/train_qwen25vl_dpo_v8_lora.py
    --config configs/dpo_v8_lingo_smoke.yaml
    --model_path "$MODEL_PATH"
    --init_adapter "$INIT_ADAPTER"
    --reference_adapter "$REFERENCE_ADAPTER"
    --preference_file data/train/preference_v8/preference_v8_pairs.jsonl
    --output_dir "$TRAIN_OUTPUT"
    --log_dir "$TRAIN_LOGS"
    --max_steps "$MAX_STEPS"
  )
  if [[ "$BF16" == "1" ]]; then TRAIN_ARGS+=(--bf16); fi
  if [[ "$QLORA" == "1" ]]; then TRAIN_ARGS+=(--qlora); fi
  if [[ "$RUN" != "1" ]]; then TRAIN_ARGS+=(--dry_run); fi
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
fi

if [[ "$DO_EVAL" == "1" ]]; then
  echo "[stage6] 3/6 held-out strict visual-control eval"
  EVAL_ARGS=(
    src/eval/run_visual_control_eval.py
    --dataset lingoqa
    --visual_control_dir data/processed/visual_control_heldout
    --eval_ids "$EVAL_IDS"
    --model_path "$MODEL_PATH"
    --adapter_path "$TRAIN_OUTPUT"
    --model_name dpo_v8_lingo_smoke
    --prediction_root "$PREDICTION_ROOT"
    --output_root "$EVAL_OUTPUT_ROOT"
    --mode strict_visual
  )
  if [[ "$BF16" == "1" ]]; then EVAL_ARGS+=(--bf16); fi
  if [[ "$RUN" != "1" ]]; then EVAL_ARGS+=(--dry_run); fi
  "$PYTHON_BIN" "${EVAL_ARGS[@]}"
fi

if [[ "$DO_RESULTS" == "1" ]]; then
  echo "[stage6] 4/6 build held-out result table"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" src/eval/build_stage6_dpo_v8_results.py \
      --prediction_root outputs/predictions_heldout \
      --output "$RESULTS_OUTPUT" \
      --warnings_output "$WARNINGS_OUTPUT"
  else
    echo "[stage6] dry-run: results builder requires formal DPO-v8 held-out predictions and is deferred."
  fi
fi

if [[ "$DO_DIAGNOSE" == "1" ]]; then
  echo "[stage6] 5/6 diagnosis"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" src/eval/diagnose_stage6_dpo_v8.py \
      --results "$RESULTS_OUTPUT" \
      --output_json "$DIAGNOSIS_JSON" \
      --output_md "$DIAGNOSIS_MD"
  else
    echo "[stage6] dry-run: diagnosis deferred until formal held-out predictions exist."
  fi
fi

if [[ "$DO_GALLERY" == "1" ]]; then
  echo "[stage6] 6/6 case gallery"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" src/eval/build_stage6_dpo_v8_case_gallery.py \
      --prediction_root outputs/predictions_heldout \
      --output_csv "$GALLERY_CSV" \
      --output_html "$GALLERY_HTML"
  else
    echo "[stage6] dry-run: case gallery deferred until formal held-out predictions exist."
  fi
fi

echo "[stage6] complete. This workflow never launches GRPO."
