#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODEL_PATH="/root/autodl-tmp/models/Qwen3-VL-4B-Instruct"
DATASET="lingoqa"
EVAL_IDS="outputs/final_report/lingoqa_heldout_ids_100.json"
MODEL_NAME="qwen3_vl_4b_base"
PREDICTION_ROOT="outputs/predictions_qwen3_vl"
BF16=0
RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --eval_ids) EVAL_IDS="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --prediction_root) PREDICTION_ROOT="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$DATASET" == "lingoqa" || "$DATASET" == "drivelm" ]] || { echo "--dataset must be lingoqa or drivelm" >&2; exit 2; }
[[ "$PREDICTION_ROOT" == outputs/predictions_qwen3_vl* ]] || { echo "Prediction root must stay under outputs/predictions_qwen3_vl" >&2; exit 2; }
VISUAL_DIR="data/processed/visual_control_heldout"
[[ "$DATASET" == "drivelm" ]] && VISUAL_DIR="data/processed/visual_control_drivelm_ood"
cmd=("$PYTHON_BIN" "src/eval/run_qwen3_vl_visual_control_eval.py" "--dataset" "$DATASET" "--visual_control_dir" "$VISUAL_DIR" "--eval_ids" "$EVAL_IDS" "--model_path" "$MODEL_PATH" "--model_name" "$MODEL_NAME" "--prediction_root" "$PREDICTION_ROOT" "--mode" "strict_visual")
[[ "$BF16" == "1" ]] && cmd+=("--bf16")
if [[ "$RUN" != "1" ]]; then
  printf '[dry-run] %q ' "${cmd[@]}"; printf '\n'
  echo "Qwen3-VL base evaluation is prepared only; no model is loaded in Stage 15."
  exit 0
fi
echo "GPU evaluation explicitly requested. This performs inference only and never trains."
"${cmd[@]}"
