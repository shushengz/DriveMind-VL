#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"

TRAIN_FILE="${TRAIN_FILE:-data/processed/drivelm_visual_pref_v2_train.jsonl}" \
MAX_PAIRS="${MAX_PAIRS:-1200}" \
MAX_STEPS="${MAX_STEPS:-48}" \
SAVE_STEPS="${SAVE_STEPS:-24}" \
LR="${LR:-2e-7}" \
DPO_BETA="${DPO_BETA:-0.10}" \
CHOSEN_SFT_WEIGHT="${CHOSEN_SFT_WEIGHT:-0.05}" \
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/qwen25vl_3b_drivelm_visual_pref_v2}" \
bash scripts/65_train_drivelm_visual_pref_v1.sh "${MODEL_DIR}"
