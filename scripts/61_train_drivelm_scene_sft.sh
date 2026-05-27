#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

python src/train_qwen25vl_lora.py \
  --model_name_or_path "${MODEL_DIR}" \
  --init_adapter_path "${INIT_ADAPTER:-}" \
  --train_file "${TRAIN_FILE:-data/processed/drivelm_train_scene.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/checkpoints/qwen25vl_3b_drivelm_scene_sft_1200}" \
  --max_samples "${MAX_SAMPLES:-1200}" \
  --epochs "${EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:-0}" \
  --save_steps "${SAVE_STEPS:-0}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --learning_rate "${LR:-8e-6}" \
  --shuffle_seed "${SHUFFLE_SEED:-20260520}" \
  --use_all_images \
  --max_images "${MAX_IMAGES:-5}" \
  --frame_strategy "${FRAME_STRATEGY:-uniform}" \
  --prompt_variant "${PROMPT_VARIANT:-spatial}" \
  --perception_mode "${PERCEPTION_MODE:-full}" \
  --max_pixels "${MAX_PIXELS:-401408}" \
  --bf16 \
  --gradient_checkpointing
