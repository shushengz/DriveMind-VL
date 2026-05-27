#!/usr/bin/env bash
set -euo pipefail

echo "LingoQA larger held-out evaluation only: inference without training. Stage 12 prepares this script but does not run it."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
MODELS="base,sft_v2,sft_v3_r3,dpo_v8_2"
EVAL_IDS="outputs/final_report/lingoqa_heldout_ids_300.json"
PREDICTION_ROOT="outputs/predictions_large_heldout"
MODEL_PATH="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
RUN=0; BF16=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --eval_ids) EVAL_IDS="$2"; shift 2 ;;
    --prediction_root) PREDICTION_ROOT="$2"; shift 2 ;;
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --bf16) BF16=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "[large-heldout] unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$PREDICTION_ROOT" != "outputs/predictions_heldout" ]] || { echo "refusing to overwrite final held-out predictions" >&2; exit 1; }
[[ -f "$EVAL_IDS" ]] || { echo "missing eval ID file: $EVAL_IDS" >&2; exit 1; }

declare -A ADAPTERS=(
  [base]=""
  [sft_v2]="checkpoints/qwen25vl_lora_sft_v2/"
  [dpo_v7]="checkpoints/qwen25vl_lora_dpo_v7/"
  [sft_v3_r3]="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
  [dpo_v8]="checkpoints/qwen25vl_lora_dpo_v8_lingo_smoke/"
  [dpo_v8_1_step25]="checkpoints/qwen25vl_lora_dpo_v8_1_lingo_smoke/checkpoint-step-000025/"
  [dpo_v8_1_step50]="checkpoints/qwen25vl_lora_dpo_v8_1_lingo_smoke/"
  [dpo_v8_2]="checkpoints/qwen25vl_lora_dpo_v8_2_lingo_smoke/"
)
declare -A NAMES=(
  [base]="base_qwen25vl_3b"
  [sft_v2]="sft_v2"
  [dpo_v7]="dpo_v7"
  [sft_v3_r3]="sft_v3_r3_lingo_smoke"
  [dpo_v8]="dpo_v8_lingo_smoke"
  [dpo_v8_1_step25]="dpo_v8_1_lingo_smoke_step25"
  [dpo_v8_1_step50]="dpo_v8_1_lingo_smoke_step50"
  [dpo_v8_2]="dpo_v8_2_lingo_smoke_step25"
)

IFS=',' read -r -a CHOSEN <<< "$MODELS"
for key in "${CHOSEN[@]}"; do
  [[ -v "NAMES[$key]" ]] || { echo "unknown model key: $key" >&2; exit 1; }
  args=(src/eval/run_visual_control_eval.py --dataset lingoqa --visual_control_dir data/processed/visual_control_large_heldout --eval_ids "$EVAL_IDS" --model_path "$MODEL_PATH" --model_name "${NAMES[$key]}" --prediction_root "$PREDICTION_ROOT" --output_root outputs/large_heldout_eval --mode strict_visual --max_new_tokens 64)
  [[ -z "${ADAPTERS[$key]}" ]] || args+=(--adapter_path "${ADAPTERS[$key]}")
  [[ "$BF16" == "1" ]] && args+=(--bf16)
  target="$PREDICTION_ROOT/lingoqa/${NAMES[$key]}/strict_visual/normal.jsonl"
  [[ "$RUN" != "1" || ! -e "$target" ]] || { echo "refusing to overwrite: $target" >&2; exit 1; }
  echo "$PYTHON_BIN ${args[*]}"
  [[ "$RUN" == "1" ]] && "$PYTHON_BIN" "${args[@]}"
done
echo "Use src/eval/build_large_heldout_results.py afterwards; final conclusions use answer-only scores."
