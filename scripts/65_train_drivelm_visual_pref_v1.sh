#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

python src/train_qwen25vl_dpo_lora.py \
  --model_name_or_path "${MODEL_DIR}" \
  --init_adapter_path "${INIT_ADAPTER:-outputs/checkpoints/qwen25vl_3b_drivelm_scene_sft_1200}" \
  --train_file "${TRAIN_FILE:-data/processed/drivelm_visual_pref_v1_train.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/checkpoints/qwen25vl_3b_drivelm_visual_pref_v1}" \
  --max_pairs "${MAX_PAIRS:-800}" \
  --epochs "${EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:-24}" \
  --save_steps "${SAVE_STEPS:-12}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --learning_rate "${LR:-1.5e-7}" \
  --loss_type "${LOSS_TYPE:-dpo}" \
  --beta "${DPO_BETA:-0.05}" \
  --chosen_sft_weight "${CHOSEN_SFT_WEIGHT:-0.10}" \
  --default_normal_sft_anchor_weight "${DEFAULT_NORMAL_SFT_ANCHOR_WEIGHT:-1.0}" \
  --default_control_sft_anchor_weight "${DEFAULT_CONTROL_SFT_ANCHOR_WEIGHT:-0.0}" \
  --shuffle_seed "${SHUFFLE_SEED:-20260521}" \
  --reference_free \
  --use_all_images \
  --max_images "${MAX_IMAGES:-5}" \
  --frame_strategy "${FRAME_STRATEGY:-uniform}" \
  --prompt_variant "${PROMPT_VARIANT:-spatial}" \
  --perception_mode "${PERCEPTION_MODE:-full}" \
  --max_pixels "${MAX_PIXELS:-401408}" \
  --bf16 \
  --gradient_checkpointing
