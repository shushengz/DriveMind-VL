#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

SPLIT="${SPLIT:-train}" \
ADAPTER="${ADAPTER:-outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v2_visual_scale}" \
PREFIX="${PREFIX:-lingoqa_clean_v2_train_sft_v2_visual_scale_for_pref}" \
MAX_SAMPLES="${MAX_SAMPLES:-1000}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}" \
MAX_PIXELS="${MAX_PIXELS:-401408}" \
FRAME_STRATEGY="${FRAME_STRATEGY:-uniform}" \
MAX_IMAGES="${MAX_IMAGES:-5}" \
PROMPT_VARIANT="${PROMPT_VARIANT:-spatial}" \
bash scripts/46_eval_lingoqa_clean_visual_controls.sh "$MODEL_DIR"
