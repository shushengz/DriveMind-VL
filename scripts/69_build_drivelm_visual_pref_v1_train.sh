#!/usr/bin/env bash
set -euo pipefail

DATA="${DATA:-data/processed/drivelm_train_scene.jsonl}" \
PREFIX="${PREFIX:-drivelm_train_scene_sft_1200}" \
OUTPUT="${OUTPUT:-data/processed/drivelm_visual_pref_v1_train.jsonl}" \
SUMMARY="${SUMMARY:-outputs/eval_results/drivelm_visual_pref_v1_train_summary.json}" \
MARKDOWN="${MARKDOWN:-docs/drivelm_visual_pref_v1_train_audit.md}" \
MAX_NORMAL_ANCHOR_PAIRS="${MAX_NORMAL_ANCHOR_PAIRS:-600}" \
MAX_FAILURE_CASES="${MAX_FAILURE_CASES:-800}" \
bash scripts/64_build_drivelm_visual_pref_v1.sh
