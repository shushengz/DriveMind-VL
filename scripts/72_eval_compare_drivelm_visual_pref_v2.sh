#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
PREFIX="${PREFIX:-drivelm_dev_scene_visual_pref_v2}"
ADAPTER="${ADAPTER:-outputs/checkpoints/qwen25vl_3b_drivelm_visual_pref_v2/checkpoint-step-000048}"

DATA="${DATA:-data/processed/drivelm_dev_scene.jsonl}" \
PREFIX="${PREFIX}" \
ADAPTER="${ADAPTER}" \
MAX_SAMPLES="${MAX_SAMPLES:-300}" \
MAX_IMAGES="${MAX_IMAGES:-5}" \
FRAME_STRATEGY="${FRAME_STRATEGY:-uniform}" \
PROMPT_VARIANT="${PROMPT_VARIANT:-spatial}" \
bash scripts/55_eval_external_visual_controls.sh "${MODEL_DIR}"

python src/eval/compare_visual_control_runs.py \
  --runs \
  "base=outputs/eval_results/${BASE_PREFIX:-drivelm_dev_scene_base}_visual_control_summary.json" \
  "sft_1200=outputs/eval_results/${SFT_PREFIX:-drivelm_dev_scene_sft_1200}_visual_control_summary.json" \
  "visual_pref_v1=outputs/eval_results/${PREF_V1_PREFIX:-drivelm_dev_scene_visual_pref_v1}_visual_control_summary.json" \
  "visual_pref_v2=outputs/eval_results/${PREFIX}_visual_control_summary.json" \
  --baseline base \
  --incumbent sft_1200 \
  --output_json "${OUTPUT_JSON:-outputs/eval_results/drivelm_visual_pref_v2_model_comparison.json}" \
  --output_md "${OUTPUT_MD:-docs/drivelm_visual_pref_v2_model_comparison.md}" \
  --title "DriveLM Visual-Preference V2 Comparison" \
  --min_normal_f1 "${MIN_NORMAL_F1:-0.35}" \
  --min_per_case_gap "${MIN_PER_CASE_GAP:--0.10}"
