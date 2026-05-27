#!/usr/bin/env bash
set -euo pipefail
case "${OMP_NUM_THREADS:-}" in ""|0|*[!0-9]*) export OMP_NUM_THREADS=1 ;; esac
case "${MKL_NUM_THREADS:-}" in ""|0|*[!0-9]*) export MKL_NUM_THREADS="$OMP_NUM_THREADS" ;; esac
RUN=0; DATASET="both"; EVAL_SAMPLES=50; TRAIN_SAMPLES=1000; MAX_STEPS=100
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"; OUTPUT_ROOT="outputs"; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
SKIP_TRAIN=0; SKIP_EVAL=0; SKIP_OLD=0; BF16=0; QLORA=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;; --dataset) DATASET="$2"; shift 2 ;; --eval_samples) EVAL_SAMPLES="$2"; shift 2 ;; --train_samples) TRAIN_SAMPLES="$2"; shift 2 ;; --max_steps) MAX_STEPS="$2"; shift 2 ;; --model_path) MODEL_PATH="$2"; shift 2 ;; --output_root) OUTPUT_ROOT="$2"; shift 2 ;; --skip_train) SKIP_TRAIN=1; shift ;; --skip_eval) SKIP_EVAL=1; shift ;; --skip_old_checkpoints) SKIP_OLD=1; shift ;; --bf16) BF16=1; shift ;; --qlora) QLORA=1; shift ;; --dry_run) RUN=0; shift ;; *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
DRY_FLAG="--dry_run"; [[ "$RUN" -eq 1 ]] && DRY_FLAG=""
BF16_FLAG=""; [[ "$BF16" -eq 1 ]] && BF16_FLAG="--bf16"
QLORA_FLAG=""; [[ "$QLORA" -eq 1 ]] && QLORA_FLAG="--qlora"
cd "$(dirname "$0")/.."; mkdir -p "$OUTPUT_ROOT/gpu_stage" "$OUTPUT_ROOT/final_report"
echo "[stage2] mode=$([[ $RUN -eq 1 ]] && echo run || echo dry_run) dataset=$DATASET eval_samples=$EVAL_SAMPLES max_steps=$MAX_STEPS"
echo "[stage2] 1/8 readiness check"
if [[ "$RUN" -eq 1 ]]; then
  "$PYTHON_BIN" src/data/visual_control_formatter.py --input_lingoqa data/processed/lingoqa_clean_v2_train.jsonl --input_drivelm data/processed/drivelm_train_scene.jsonl --output_dir data/processed/visual_control --blank_image outputs/cases/ablation_images/blank.jpg
  "$PYTHON_BIN" src/data/build_sft_v3.py --input_lingoqa data/processed/visual_control/lingoqa_strict_normal.jsonl --input_drivelm data/processed/visual_control/drivelm_strict_normal.jsonl --output_dir data/train/sft_v3
  "$PYTHON_BIN" src/data/build_preference_v8.py --sft_v3_data data/train/sft_v3/mixed_sft_v3.jsonl --output_dir data/train/preference_v8
fi
bash scripts/check_gpu_stage_ready.sh --python "$PYTHON_BIN" --model_path "$MODEL_PATH" --output_dir "$OUTPUT_ROOT/gpu_stage"
if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo "[stage2] 2/8 base strict eval"
  "$PYTHON_BIN" src/eval/run_visual_control_eval.py $DRY_FLAG --dataset "$DATASET" --eval_samples "$EVAL_SAMPLES" --model_path "$MODEL_PATH" --model_name base_qwen25vl_3b --output_root "$OUTPUT_ROOT" $BF16_FLAG
fi
if [[ "$SKIP_OLD" -eq 0 && "$SKIP_EVAL" -eq 0 ]]; then
  echo "[stage2] 3/8 old checkpoint regression eval"
  "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
candidates={"sft_v2":["outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale","checkpoints/qwen25vl_lora_sft_v2"],"dpo_v7":["outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v7_grounded","checkpoints/qwen25vl_lora_dpo_v7"]}
missing={k:v for k,v in candidates.items() if not any(Path(p).exists() for p in v)}
Path("outputs/gpu_stage").mkdir(parents=True,exist_ok=True); Path("outputs/gpu_stage/missing_checkpoints.json").write_text(json.dumps(missing,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps({"missing":missing},ensure_ascii=False))
PY
  for spec in "sft_v2:outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale" "dpo_v7:outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v7_grounded"; do
    name="${spec%%:*}"; adapter="${spec#*:}"
    if [[ -d "$adapter" ]]; then "$PYTHON_BIN" src/eval/run_visual_control_eval.py $DRY_FLAG --dataset "$DATASET" --eval_samples "$EVAL_SAMPLES" --model_path "$MODEL_PATH" --adapter_path "$adapter" --model_name "$name" --output_root "$OUTPUT_ROOT" $BF16_FLAG; fi
  done
fi
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  echo "[stage2] 4/8 SFT-v3 lingo smoke train"
  "$PYTHON_BIN" src/train/train_qwen25vl_sft_v3_lora.py $DRY_FLAG --model_path "$MODEL_PATH" --train_file data/train/sft_v3/lingoqa_sft_v3.jsonl --output_dir checkpoints/qwen25vl_lora_sft_v3_lingo_smoke --log_dir "$OUTPUT_ROOT/train_logs/sft_v3_lingo_smoke" --train_samples "$TRAIN_SAMPLES" --max_steps "$MAX_STEPS" --save_steps 100 --gradient_accumulation_steps 8 --learning_rate 1e-5 $BF16_FLAG $QLORA_FLAG
fi
if [[ "$SKIP_EVAL" -eq 0 ]]; then
  echo "[stage2] 5/8 SFT-v3 strict eval"
  adapter="checkpoints/qwen25vl_lora_sft_v3_lingo_smoke"
  if [[ "$RUN" -eq 0 || -f "$adapter/adapter_config.json" ]]; then "$PYTHON_BIN" src/eval/run_visual_control_eval.py $DRY_FLAG --dataset lingoqa --eval_samples "$EVAL_SAMPLES" --model_path "$MODEL_PATH" --adapter_path "$adapter" --model_name sft_v3_lingo_smoke --output_root "$OUTPUT_ROOT" $BF16_FLAG; else echo "[stage2] skip SFT-v3 eval: valid adapter_config.json not found in $adapter"; fi
fi
echo "[stage2] 6/8 collect summaries"; "$PYTHON_BIN" src/eval/collect_stage2_results.py --eval_results_dir "$OUTPUT_ROOT/eval_results" --output "$OUTPUT_ROOT/final_report/stage2_strict_eval_main_results.csv"
echo "[stage2] 7/8 diagnose"; "$PYTHON_BIN" src/eval/diagnose_stage2_results.py --input "$OUTPUT_ROOT/final_report/stage2_strict_eval_main_results.csv" --output_json "$OUTPUT_ROOT/final_report/stage2_diagnosis.json" --output_md "$OUTPUT_ROOT/final_report/stage2_diagnosis.md"
echo "[stage2] 8/8 case gallery"; case_file="$OUTPUT_ROOT/cases/lingoqa_sft_v3_lingo_smoke_strict_visual_case_scores.csv"
if [[ -f "$case_file" ]]; then "$PYTHON_BIN" src/eval/build_case_gallery.py --case_scores "$case_file" --visual_samples data/processed/visual_control/lingoqa_strict_normal.jsonl --output_dir "$OUTPUT_ROOT/final_report" $DRY_FLAG; cp "$OUTPUT_ROOT/final_report/case_gallery.csv" "$OUTPUT_ROOT/final_report/stage2_case_gallery.csv"; cp "$OUTPUT_ROOT/final_report/case_gallery.html" "$OUTPUT_ROOT/final_report/stage2_case_gallery.html"; else echo "[stage2] skip gallery: $case_file not found"; fi
echo "[stage2] done"
