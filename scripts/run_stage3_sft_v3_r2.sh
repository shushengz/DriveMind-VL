#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS:-0}" -eq 0 ]]; then export OMP_NUM_THREADS=1; fi
if [[ -z "${MKL_NUM_THREADS:-}" || ! "${MKL_NUM_THREADS:-}" =~ ^[0-9]+$ || "${MKL_NUM_THREADS:-0}" -eq 0 ]]; then export MKL_NUM_THREADS="$OMP_NUM_THREADS"; fi

RUN=0; DO_BUILD=0; DO_AUDIT=0; DO_TRAIN=0; DO_EVAL=0; DO_DIAG=0; DO_ALL=0
DATASET="lingoqa"; MAX_TRAIN_SAMPLES=1500; MAX_EVAL_SAMPLES=100; MAX_STEPS=200
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"; OUTPUT_ROOT="outputs"; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
BF16=0; QLORA=0; RERUN_BASELINES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;; --dry_run) RUN=0; shift ;;
    --build_data) DO_BUILD=1; shift ;; --audit_only) DO_AUDIT=1; shift ;; --train) DO_TRAIN=1; shift ;; --eval) DO_EVAL=1; shift ;; --diagnose) DO_DIAG=1; shift ;; --all) DO_ALL=1; shift ;;
    --dataset) DATASET="$2"; shift 2 ;; --max_train_samples) MAX_TRAIN_SAMPLES="$2"; shift 2 ;; --max_eval_samples) MAX_EVAL_SAMPLES="$2"; shift 2 ;; --max_steps) MAX_STEPS="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;; --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;; --qlora) QLORA=1; shift ;; --rerun_baselines) RERUN_BASELINES=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_ALL" -eq 1 ]]; then DO_BUILD=1; DO_AUDIT=1; DO_TRAIN=1; DO_EVAL=1; DO_DIAG=1; fi
if [[ "$DO_BUILD$DO_AUDIT$DO_TRAIN$DO_EVAL$DO_DIAG" == "00000" ]]; then DO_ALL=1; DO_BUILD=1; DO_AUDIT=1; DO_TRAIN=1; DO_EVAL=1; DO_DIAG=1; fi
DRY_FLAG="--dry_run"; [[ "$RUN" -eq 1 ]] && DRY_FLAG=""
BF16_FLAG=""; [[ "$BF16" -eq 1 ]] && BF16_FLAG="--bf16"
QLORA_FLAG=""; [[ "$QLORA" -eq 1 ]] && QLORA_FLAG="--qlora"
cd "$(dirname "$0")/.."
mkdir -p "$OUTPUT_ROOT/final_report" "$OUTPUT_ROOT/data_audit"
echo "[stage3] mode=$([[ $RUN -eq 1 ]] && echo run || echo dry_run) dataset=$DATASET eval=$MAX_EVAL_SAMPLES train_samples=$MAX_TRAIN_SAMPLES steps=$MAX_STEPS"

if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "[stage3] 1/build SFT-v3-r2 data"
  "$PYTHON_BIN" src/data/build_sft_v3_r2.py $DRY_FLAG --visual_control_dir data/processed/visual_control --output_dir data/train/sft_v3_r2
fi

if [[ "$DO_AUDIT" -eq 1 ]]; then
  echo "[stage3] 2/audit SFT-v3-r2 data"
  "$PYTHON_BIN" src/data/audit_sft_v3_r2.py --input data/train/sft_v3_r2/lingoqa_sft_v3_r2.jsonl --output_json "$OUTPUT_ROOT/data_audit/sft_v3_r2_lingo_audit.json" --output_md "$OUTPUT_ROOT/data_audit/sft_v3_r2_lingo_audit.md"
  if [[ "$DO_AUDIT" -eq 1 && "$DO_TRAIN$DO_EVAL$DO_DIAG" == "000" ]]; then exit 0; fi
fi

if [[ "$DO_TRAIN" -eq 1 ]]; then
  echo "[stage3] 3/train SFT-v3-r2 lingo smoke"
  "$PYTHON_BIN" src/train/train_qwen25vl_sft_v3_lora.py $DRY_FLAG --model_path "$MODEL_PATH" --train_file data/train/sft_v3_r2/lingoqa_sft_v3_r2.jsonl --output_dir checkpoints/qwen25vl_lora_sft_v3_r2_lingo_smoke --log_dir "$OUTPUT_ROOT/train_logs/sft_v3_r2_lingo_smoke" --train_samples "$MAX_TRAIN_SAMPLES" --max_steps "$MAX_STEPS" --save_steps 100 --eval_steps 50 --eval_samples 100 --gradient_accumulation_steps 8 --learning_rate 1e-5 --max_pixels 200704 --max_images 3 $BF16_FLAG $QLORA_FLAG
fi

if [[ "$DO_EVAL" -eq 1 ]]; then
  echo "[stage3] 4/eval SFT-v3-r2"
  adapter="checkpoints/qwen25vl_lora_sft_v3_r2_lingo_smoke"
  if [[ "$RUN" -eq 0 || -f "$adapter/adapter_config.json" ]]; then
    "$PYTHON_BIN" src/eval/run_visual_control_eval.py $DRY_FLAG --dataset lingoqa --eval_samples "$MAX_EVAL_SAMPLES" --model_path "$MODEL_PATH" --adapter_path "$adapter" --model_name sft_v3_r2_lingo_smoke --output_root "$OUTPUT_ROOT" $BF16_FLAG
  else
    echo "[stage3] missing adapter_config.json in $adapter" >&2; exit 1
  fi
  if [[ "$RERUN_BASELINES" -eq 1 ]]; then
    echo "[stage3] 4b/rerun fair baselines"
    "$PYTHON_BIN" src/eval/run_visual_control_eval.py $DRY_FLAG --dataset lingoqa --eval_samples "$MAX_EVAL_SAMPLES" --model_path "$MODEL_PATH" --model_name base_qwen25vl_3b --output_root "$OUTPUT_ROOT" $BF16_FLAG
    for spec in "sft_v2:outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale" "dpo_v7:outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v7_grounded" "sft_v3_lingo_smoke:checkpoints/qwen25vl_lora_sft_v3_lingo_smoke"; do
      name="${spec%%:*}"; ckpt="${spec#*:}"
      if [[ -f "$ckpt/adapter_config.json" ]]; then
        "$PYTHON_BIN" src/eval/run_visual_control_eval.py $DRY_FLAG --dataset lingoqa --eval_samples "$MAX_EVAL_SAMPLES" --model_path "$MODEL_PATH" --adapter_path "$ckpt" --model_name "$name" --output_root "$OUTPUT_ROOT" $BF16_FLAG
      else
        echo "[stage3] skip baseline $name: no adapter_config.json in $ckpt"
      fi
    done
  fi
fi

if [[ "$DO_DIAG" -eq 1 ]]; then
  echo "[stage3] 5/collect stage3 main results"
  "$PYTHON_BIN" src/eval/collect_stage2_results.py --eval_results_dir "$OUTPUT_ROOT/eval_results" --output "$OUTPUT_ROOT/final_report/stage3_sft_v3_r2_main_results.csv" --include_models base_qwen25vl_3b,sft_v2,dpo_v7,sft_v3_lingo_smoke,sft_v3_r2_lingo_smoke --min_num_samples 30
  echo "[stage3] 6/diagnose"
  "$PYTHON_BIN" src/eval/diagnose_stage3_sft_v3_r2.py --input "$OUTPUT_ROOT/final_report/stage3_sft_v3_r2_main_results.csv" --output_json "$OUTPUT_ROOT/final_report/stage3_sft_v3_r2_diagnosis.json" --output_md "$OUTPUT_ROOT/final_report/stage3_sft_v3_r2_diagnosis.md"
  echo "[stage3] 7/case gallery"
  if [[ -f "$OUTPUT_ROOT/cases/lingoqa_sft_v3_r2_lingo_smoke_strict_visual_case_scores.csv" ]]; then
    "$PYTHON_BIN" src/eval/build_stage3_sft_v3_r2_case_gallery.py --output_csv "$OUTPUT_ROOT/final_report/stage3_sft_v3_r2_case_gallery.csv" --output_html "$OUTPUT_ROOT/final_report/stage3_sft_v3_r2_case_gallery.html"
  else
    echo "[stage3] skip gallery: r2 case scores not found"
  fi
fi

echo "[stage3] done"
