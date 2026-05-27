#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
EVAL_IDS="outputs/final_report/drivelm_ood_ids_100.json"
PREDICTION_ROOT="outputs/predictions_drivelm_ood"
OUTPUT_ROOT="outputs/drivelm_ood_eval"
RUN=0
BF16=0
DO_CHECK=0
DO_BASE=0
DO_R3=0
DO_RESULTS=0
DO_CAPABILITY=0
DO_DIAGNOSE=0
DO_GALLERY=0
DO_REPORT=0

usage() {
  cat <<'EOF'
Stage 13 DriveLM OOD strict visual-control GPU evaluation.
Inference only: this script never trains SFT/DPO/GRPO and never constructs training data.
Default mode is dry-run. Pass --run explicitly to execute GPU inference or analysis.

Options:
  --check --eval_base --eval_r3 --results --capability --diagnose --gallery --update_report --all
  --run --dry_run --model_path PATH --eval_ids PATH --bf16
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check) DO_CHECK=1; shift ;;
    --eval_base) DO_BASE=1; shift ;;
    --eval_r3) DO_R3=1; shift ;;
    --results) DO_RESULTS=1; shift ;;
    --capability) DO_CAPABILITY=1; shift ;;
    --diagnose) DO_DIAGNOSE=1; shift ;;
    --gallery) DO_GALLERY=1; shift ;;
    --update_report) DO_REPORT=1; shift ;;
    --all) DO_CHECK=1; DO_BASE=1; DO_R3=1; DO_RESULTS=1; DO_CAPABILITY=1; DO_DIAGNOSE=1; DO_GALLERY=1; DO_REPORT=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --eval_ids) EVAL_IDS="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[stage13] unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ "$DO_CHECK$DO_BASE$DO_R3$DO_RESULTS$DO_CAPABILITY$DO_DIAGNOSE$DO_GALLERY$DO_REPORT" == "00000000" ]]; then
  usage
  exit 0
fi

echo "Stage 13: DriveLM OOD strict visual-control evaluation only. No training, no GRPO, no LingoQA overwrite."
[[ "$PREDICTION_ROOT" == "outputs/predictions_drivelm_ood" ]] || { echo "[stage13] isolated prediction root required" >&2; exit 1; }
[[ "$EVAL_IDS" == outputs/final_report/drivelm_ood_ids_100.json || "$EVAL_IDS" == outputs/final_report/drivelm_ood_ids_300.json ]] || {
  echo "[stage13] eval IDs must be an approved DriveLM OOD ID set" >&2; exit 1;
}
if [[ "$EVAL_IDS" == *"_300.json" && ( "$DO_BASE" == "1" || "$DO_R3" == "1" ) ]]; then
  echo "[stage13] 300-case inference is not authorized in this stage; use the 100-case smoke IDs." >&2
  exit 1
fi

bf16_args=()
[[ "$BF16" == "1" ]] && bf16_args+=(--bf16)
run_log="outputs/gpu_stage/stage13_drivelm_ood_run.log"
mkdir -p outputs/gpu_stage

check_command=(bash scripts/check_stage13_drivelm_ood_ready.sh "$MODEL_PATH" "$EVAL_IDS")
eval_base_command=("$PYTHON_BIN" src/eval/run_visual_control_eval.py --dataset drivelm --visual_control_dir data/processed/visual_control --eval_ids "$EVAL_IDS" --model_path "$MODEL_PATH" --model_name base_qwen25vl_3b --prediction_root "$PREDICTION_ROOT" --output_root "$OUTPUT_ROOT" --mode strict_visual --max_new_tokens 64 "${bf16_args[@]}")
eval_r3_command=("$PYTHON_BIN" src/eval/run_visual_control_eval.py --dataset drivelm --visual_control_dir data/processed/visual_control --eval_ids "$EVAL_IDS" --model_path "$MODEL_PATH" --adapter_path checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/ --model_name sft_v3_r3_lingo_smoke --prediction_root "$PREDICTION_ROOT" --output_root "$OUTPUT_ROOT" --mode strict_visual --max_new_tokens 64 "${bf16_args[@]}")

if [[ "$RUN" != "1" ]]; then
  [[ "$DO_CHECK" == "1" || "$DO_BASE" == "1" || "$DO_R3" == "1" ]] && printf '[dry-run] %q ' "${check_command[@]}" && echo
  [[ "$DO_BASE" == "1" ]] && printf '[dry-run] %q ' "${eval_base_command[@]}" && echo
  [[ "$DO_R3" == "1" ]] && printf '[dry-run] %q ' "${eval_r3_command[@]}" && echo
  [[ "$DO_RESULTS" == "1" ]] && echo "[dry-run] $PYTHON_BIN src/eval/build_drivelm_ood_results.py"
  [[ "$DO_CAPABILITY" == "1" ]] && echo "[dry-run] $PYTHON_BIN src/eval/analyze_drivelm_ood_capability.py"
  [[ "$DO_DIAGNOSE" == "1" ]] && echo "[dry-run] $PYTHON_BIN src/eval/diagnose_drivelm_ood_results.py"
  [[ "$DO_GALLERY" == "1" ]] && echo "[dry-run] $PYTHON_BIN src/eval/build_drivelm_ood_case_gallery.py"
  [[ "$DO_REPORT" == "1" ]] && echo "[dry-run] $PYTHON_BIN src/eval/update_stage13_drivelm_ood_report.py"
  exit 0
fi

if [[ "$DO_CHECK" == "1" || "$DO_BASE" == "1" || "$DO_R3" == "1" ]]; then
  "${check_command[@]}"
fi

run_eval() {
  local model_name="$1"; shift
  local target="$PREDICTION_ROOT/drivelm/$model_name/strict_visual"
  if find "$target" -maxdepth 1 -name '*.jsonl' -print -quit 2>/dev/null | grep -q .; then
    echo "[stage13] refusing to overwrite existing DriveLM predictions in $target" >&2
    exit 1
  fi
  echo "[stage13] inference start: $model_name" | tee -a "$run_log"
  set +e
  "$@" 2>&1 | tee -a "$run_log"
  status=${PIPESTATUS[0]}
  set -e
  if [[ "$status" -ne 0 ]]; then
    if grep -Eiq 'out of memory|CUDA error|CUDNN_STATUS_ALLOC_FAILED' "$run_log"; then
      echo "[stage13] GPU OOM/CUDA failure recorded; stopping without further inference." | tee -a "$run_log"
    fi
    exit "$status"
  fi
  echo "[stage13] inference complete: $model_name" | tee -a "$run_log"
}

[[ "$DO_BASE" == "1" ]] && run_eval base_qwen25vl_3b "${eval_base_command[@]}"
[[ "$DO_R3" == "1" ]] && run_eval sft_v3_r3_lingo_smoke "${eval_r3_command[@]}"
[[ "$DO_RESULTS" == "1" ]] && "$PYTHON_BIN" src/eval/build_drivelm_ood_results.py
[[ "$DO_CAPABILITY" == "1" ]] && "$PYTHON_BIN" src/eval/analyze_drivelm_ood_capability.py
[[ "$DO_DIAGNOSE" == "1" ]] && "$PYTHON_BIN" src/eval/diagnose_drivelm_ood_results.py
[[ "$DO_GALLERY" == "1" ]] && "$PYTHON_BIN" src/eval/build_drivelm_ood_case_gallery.py
[[ "$DO_REPORT" == "1" ]] && "$PYTHON_BIN" src/eval/update_stage13_drivelm_ood_report.py

echo "[stage13] selected actions finished. Any GPU computation in this workflow is limited to requested DriveLM OOD inference; no training or GRPO action was executed."
