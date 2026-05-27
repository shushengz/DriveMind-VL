#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

DATA="${DATA:-data/processed/drivelm_dev_scene.jsonl}" \
PREFIX="${PREFIX:-drivelm_dev_scene_visual_pref_v1}" \
ADAPTER="${ADAPTER:-outputs/checkpoints/qwen25vl_3b_drivelm_visual_pref_v1/checkpoint-step-000024}" \
MAX_SAMPLES="${MAX_SAMPLES:-300}" \
MAX_IMAGES="${MAX_IMAGES:-5}" \
FRAME_STRATEGY="${FRAME_STRATEGY:-uniform}" \
PROMPT_VARIANT="${PROMPT_VARIANT:-spatial}" \
bash scripts/55_eval_external_visual_controls.sh "${MODEL_DIR}"
