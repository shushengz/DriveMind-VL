#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

python src/train_qwen25vl_dpo_lora.py \
  --model_name_or_path "$MODEL_DIR" \
  --train_file "${TRAIN_FILE:-data/processed/lingoqa_sft_v1_preference_pairs_77.jsonl}" \
  --output_dir "${OUTPUT_DIR:-outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v1_dpo_pref77_smoke}" \
  --max_pairs "${MAX_PAIRS:-0}" \
  --epochs "${EPOCHS:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --learning_rate "${LR:-5e-6}" \
  --beta "${DPO_BETA:-0.1}" \
  --chosen_sft_weight "${CHOSEN_SFT_WEIGHT:-0.05}" \
  --shuffle_seed "${SHUFFLE_SEED:-20260517}" \
  --use_all_images \
  --max_images "${MAX_IMAGES:-3}" \
  --frame_strategy "${FRAME_STRATEGY:-first_middle_last}" \
  --prompt_variant "${PROMPT_VARIANT:-spatial}" \
  --max_pixels "${MAX_PIXELS:-200704}" \
  --bf16 \
  --gradient_checkpointing
