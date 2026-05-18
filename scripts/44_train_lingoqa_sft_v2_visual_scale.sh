#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

python src/train_qwen25vl_lora.py \
  --model_name_or_path "$MODEL_DIR" \
  --train_file "${TRAIN_FILE:-data/processed/lingoqa_sft_v2_visual_train.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale}" \
  --max_samples "${MAX_SAMPLES:-5000}" \
  --epochs "${EPOCHS:-2}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --learning_rate "${LR:-1e-5}" \
  --shuffle_seed "${SHUFFLE_SEED:-20260517}" \
  --use_all_images \
  --max_images "${MAX_IMAGES:-5}" \
  --frame_strategy "${FRAME_STRATEGY:-uniform}" \
  --prompt_variant "${PROMPT_VARIANT:-spatial}" \
  --max_pixels "${MAX_PIXELS:-401408}" \
  --bf16 \
  --gradient_checkpointing
