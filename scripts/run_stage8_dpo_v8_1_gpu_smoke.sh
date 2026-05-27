#!/usr/bin/env bash
set -euo pipefail

echo "GPU Stage 8 smoke: model-mined DPO-v8.1 only. No GRPO and no large-scale training."
export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
INIT_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
REFERENCE_ADAPTER="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
EVAL_IDS="outputs/final_report/stage4_5_heldout_ids_100.json"
MAX_STEPS=50
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
    --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --eval_ids) EVAL_IDS="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --qlora) QLORA=1; shift ;;
    *) echo "[stage8] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_CHECK$DO_TRAIN$DO_EVAL$DO_RESULTS$DO_DIAGNOSE$DO_GALLERY" == "000000" ]]; then
  DO_CHECK=1; DO_TRAIN=1; DO_EVAL=1; DO_RESULTS=1; DO_DIAGNOSE=1; DO_GALLERY=1
fi
MODE="dry_run"; [[ "$RUN" == "1" ]] && MODE="run"
echo "[stage8] mode=${MODE} max_steps=${MAX_STEPS} eval_ids=${EVAL_IDS}"

if [[ "$RUN" == "1" ]]; then
  PRED_ROOT="outputs/predictions_heldout"; EVAL_ROOT="outputs"
  TRAIN_OUTPUT="checkpoints/qwen25vl_lora_dpo_v8_1_lingo_smoke/"; TRAIN_LOGS="outputs/train_logs/dpo_v8_1_lingo_smoke/"
  RESULTS="outputs/final_report/stage8_dpo_v8_1_heldout_main_results.csv"; WARNINGS="outputs/final_report/stage8_dpo_v8_1_heldout_main_results_warnings.json"
  DIAG_JSON="outputs/final_report/stage8_dpo_v8_1_diagnosis.json"; DIAG_MD="outputs/final_report/stage8_dpo_v8_1_diagnosis.md"
  GALLERY_CSV="outputs/final_report/stage8_dpo_v8_1_case_gallery.csv"; GALLERY_HTML="outputs/final_report/stage8_dpo_v8_1_case_gallery.html"
else
  PRED_ROOT="outputs/stage8_dry_run/predictions_heldout"; EVAL_ROOT="outputs/stage8_dry_run"
  TRAIN_OUTPUT="outputs/stage8_dry_run/checkpoints/qwen25vl_lora_dpo_v8_1_lingo_smoke/"; TRAIN_LOGS="outputs/stage8_dry_run/train_logs/dpo_v8_1_lingo_smoke/"
  RESULTS="outputs/stage8_dry_run/stage8_dpo_v8_1_heldout_main_results.csv"; WARNINGS="outputs/stage8_dry_run/stage8_dpo_v8_1_heldout_main_results_warnings.json"
  DIAG_JSON="outputs/stage8_dry_run/stage8_dpo_v8_1_diagnosis.json"; DIAG_MD="outputs/stage8_dry_run/stage8_dpo_v8_1_diagnosis.md"
  GALLERY_CSV="outputs/stage8_dry_run/stage8_dpo_v8_1_case_gallery.csv"; GALLERY_HTML="outputs/stage8_dry_run/stage8_dpo_v8_1_case_gallery.html"
fi

echo "[stage8] 1/7 readiness"
MODEL_PATH="$MODEL_PATH" PYTHON_BIN="$PYTHON_BIN" bash scripts/check_stage8_dpo_v8_1_ready.sh
if [[ "$DO_TRAIN" == "1" ]]; then
  echo "[stage8] 2/7 DPO-v8.1 smoke train"
  TRAIN_ARGS=(src/train/train_qwen25vl_dpo_v8_lora.py --config configs/dpo_v8_1_lingo_smoke.yaml --model_path "$MODEL_PATH" --init_adapter "$INIT_ADAPTER" --reference_adapter "$REFERENCE_ADAPTER" --preference_file data/train/preference_v8_1/preference_v8_1_pairs.jsonl --output_dir "$TRAIN_OUTPUT" --log_dir "$TRAIN_LOGS" --max_steps "$MAX_STEPS" --save_steps 25)
  [[ "$BF16" == "1" ]] && TRAIN_ARGS+=(--bf16)
  [[ "$QLORA" == "1" ]] && TRAIN_ARGS+=(--qlora)
  [[ "$RUN" != "1" ]] && TRAIN_ARGS+=(--dry_run)
  "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
fi
if [[ "$DO_EVAL" == "1" ]]; then
  echo "[stage8] 3/7 held-out eval step-25 and step-50"
  if [[ "$RUN" == "1" ]]; then
    for name in dpo_v8_1_lingo_smoke_step25 dpo_v8_1_lingo_smoke_step50; do
      [[ ! -e "$PRED_ROOT/lingoqa/$name/strict_visual/normal.jsonl" ]] || { echo "[stage8] refusing to overwrite existing predictions: $name" >&2; exit 1; }
    done
  fi
  STEP25="$TRAIN_OUTPUT/checkpoint-step-000025/"; STEP50="$TRAIN_OUTPUT"
  [[ "$RUN" != "1" ]] || [[ -e "$STEP25/adapter_config.json" ]] || { echo "[stage8] missing step-25 checkpoint: $STEP25" >&2; exit 1; }
  for adapter_and_name in "$STEP25:dpo_v8_1_lingo_smoke_step25" "$STEP50:dpo_v8_1_lingo_smoke_step50"; do
    adapter="${adapter_and_name%%:*}"; name="${adapter_and_name##*:}"
    EVAL_ARGS=(src/eval/run_visual_control_eval.py --dataset lingoqa --visual_control_dir data/processed/visual_control_heldout --eval_ids "$EVAL_IDS" --model_path "$MODEL_PATH" --adapter_path "$adapter" --model_name "$name" --prediction_root "$PRED_ROOT" --output_root "$EVAL_ROOT" --mode strict_visual)
    [[ "$BF16" == "1" ]] && EVAL_ARGS+=(--bf16)
    [[ "$RUN" != "1" ]] && EVAL_ARGS+=(--dry_run)
    "$PYTHON_BIN" "${EVAL_ARGS[@]}"
  done
fi
if [[ "$DO_RESULTS" == "1" ]]; then
  echo "[stage8] 5/7 build held-out results"
  if [[ "$RUN" == "1" ]]; then "$PYTHON_BIN" src/eval/build_stage8_dpo_v8_1_results.py --prediction_root outputs/predictions_heldout --output "$RESULTS" --warnings_output "$WARNINGS"; else echo "[stage8] dry-run: results deferred."; fi
fi
if [[ "$DO_DIAGNOSE" == "1" ]]; then
  echo "[stage8] 6/7 diagnosis"
  if [[ "$RUN" == "1" ]]; then "$PYTHON_BIN" src/eval/diagnose_stage8_dpo_v8_1.py --results "$RESULTS" --train_summary "$TRAIN_LOGS/train_summary.json" --output_json "$DIAG_JSON" --output_md "$DIAG_MD"; else echo "[stage8] dry-run: diagnosis deferred."; fi
fi
if [[ "$DO_GALLERY" == "1" ]]; then
  echo "[stage8] 7/7 case gallery"
  if [[ "$RUN" == "1" ]]; then "$PYTHON_BIN" src/eval/build_stage8_dpo_v8_1_case_gallery.py --prediction_root outputs/predictions_heldout --diagnosis "$DIAG_JSON" --output_csv "$GALLERY_CSV" --output_html "$GALLERY_HTML"; else echo "[stage8] dry-run: gallery deferred."; fi
fi
echo "[stage8] complete. This workflow never launches GRPO."
