#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

python src/train_qwen25vl_lora.py \
  --model_name_or_path "$MODEL_DIR" \
  --train_file data/processed/drivemind_train.jsonl \
  --output_dir outputs/checkpoints/qwen25vl_3b_lora_smoke \
  --max_samples "${MAX_SAMPLES:-80}" \
  --epochs "${EPOCHS:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --learning_rate "${LR:-1e-4}" \
  --bf16 \
  --gradient_checkpointing

