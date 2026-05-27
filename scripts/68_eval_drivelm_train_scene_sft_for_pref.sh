#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

DATA="${DATA:-data/processed/drivelm_train_scene.jsonl}" \
PREFIX="${PREFIX:-drivelm_train_scene_sft_1200}" \
ADAPTER="${ADAPTER:-outputs/checkpoints/qwen25vl_3b_drivelm_scene_sft_1200}" \
MAX_SAMPLES="${MAX_SAMPLES:-1200}" \
MAX_IMAGES="${MAX_IMAGES:-5}" \
FRAME_STRATEGY="${FRAME_STRATEGY:-uniform}" \
PROMPT_VARIANT="${PROMPT_VARIANT:-spatial}" \
bash scripts/55_eval_external_visual_controls.sh "${MODEL_DIR}"
