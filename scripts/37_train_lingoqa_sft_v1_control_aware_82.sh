#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

python src/train_qwen25vl_lora.py \
  --model_name_or_path "$MODEL_DIR" \
  --train_file data/processed/lingoqa_sft_v1_control_aware_82.jsonl \
  --output_dir outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v1_control_aware_82_smoke \
  --max_samples "${MAX_SAMPLES:-82}" \
  --epochs "${EPOCHS:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --learning_rate "${LR:-1e-5}" \
  --use_all_images \
  --max_images "${MAX_IMAGES:-3}" \
  --frame_strategy "${FRAME_STRATEGY:-first_middle_last}" \
  --prompt_variant "${PROMPT_VARIANT:-spatial}" \
  --max_pixels "${MAX_PIXELS:-200704}" \
  --bf16 \
  --gradient_checkpointing
