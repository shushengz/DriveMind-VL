#!/usr/bin/env bash
set -euo pipefail

RUN=0
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
RAW_PREDICTIONS="${RAW_PREDICTIONS:-}"
CASE_SCORES="${CASE_SCORES:-}"

for arg in "$@"; do
  case "$arg" in
    --run) RUN=1 ;;
    --raw_predictions=*) RAW_PREDICTIONS="${arg#*=}" ;;
    --case_scores=*) CASE_SCORES="${arg#*=}" ;;
    --python=*) PYTHON_BIN="${arg#*=}" ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [ "$RUN" -eq 1 ]; then
  DRY_FLAG=""
  echo "[cpu-stage] full file generation enabled (--run)."
else
  DRY_FLAG="--dry_run"
  echo "[cpu-stage] defaulting to dry-run. Pass --run to generate full files."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[cpu-stage] CUDA device may exist, but this script is CPU-only and will not use it."
else
  echo "[cpu-stage] no CUDA check available; continuing CPU-only."
fi
export CUDA_VISIBLE_DEVICES=""
cd "$(dirname "$0")/.."

echo "[cpu-stage] 1/7 build visual-control data"
"$PYTHON_BIN" src/data/visual_control_formatter.py $DRY_FLAG --input_lingoqa data/processed/lingoqa_clean_v2_train.jsonl --input_drivelm data/processed/drivelm_train_scene.jsonl --output_dir data/processed/visual_control

echo "[cpu-stage] 2/7 build SFT-v3 data"
"$PYTHON_BIN" src/data/build_sft_v3.py $DRY_FLAG --input_lingoqa data/processed/visual_control/lingoqa_strict_normal.jsonl --input_drivelm data/processed/visual_control/drivelm_strict_normal.jsonl --output_dir data/train/sft_v3

echo "[cpu-stage] 3/7 build Preference-v8 data"
PREF_ARGS=(src/data/build_preference_v8.py $DRY_FLAG --sft_v3_data data/train/sft_v3/mixed_sft_v3.jsonl --output_dir data/train/preference_v8)
if [ -n "$CASE_SCORES" ] && [ -f "$CASE_SCORES" ]; then PREF_ARGS+=(--case_scores "$CASE_SCORES"); fi
"$PYTHON_BIN" "${PREF_ARGS[@]}"

echo "[cpu-stage] 4/7 summarize visual-control predictions if provided"
if [ -n "$RAW_PREDICTIONS" ] && [ -f "$RAW_PREDICTIONS" ]; then
  "$PYTHON_BIN" src/eval/summarize_visual_control.py $DRY_FLAG --input "$RAW_PREDICTIONS" --output_dir outputs
else
  echo "[cpu-stage] skip summary: set RAW_PREDICTIONS or --raw_predictions=/path/to/raw.jsonl"
fi

echo "[cpu-stage] 5/7 run offline reward harness if predictions/cases exist"
if [ -n "$RAW_PREDICTIONS" ] && [ -f "$RAW_PREDICTIONS" ]; then
  "$PYTHON_BIN" src/rl/debug_reward_harness.py $DRY_FLAG --raw_predictions "$RAW_PREDICTIONS" --output_dir outputs/rl_debug
elif [ -n "$CASE_SCORES" ] && [ -f "$CASE_SCORES" ]; then
  "$PYTHON_BIN" src/rl/debug_reward_harness.py $DRY_FLAG --case_scores "$CASE_SCORES" --output_dir outputs/rl_debug
else
  echo "[cpu-stage] skip reward harness: no raw predictions or case scores supplied"
fi

echo "[cpu-stage] 6/7 build case gallery if case scores exist"
if [ -n "$CASE_SCORES" ] && [ -f "$CASE_SCORES" ]; then
  "$PYTHON_BIN" src/eval/build_case_gallery.py $DRY_FLAG --case_scores "$CASE_SCORES" --visual_samples data/processed/visual_control/lingoqa_strict_normal.jsonl data/processed/visual_control/drivelm_strict_normal.jsonl --output_dir outputs/final_report
else
  echo "[cpu-stage] skip case gallery: set CASE_SCORES or --case_scores=/path/to/case_scores.csv"
fi

echo "[cpu-stage] 7/7 run pytest"
"$PYTHON_BIN" -m pytest tests/test_visual_control_formatter.py tests/test_metrics_visual_control.py tests/test_build_sft_v3.py tests/test_build_preference_v8.py tests/test_reward_driving_vqa.py -q

echo "[cpu-stage] done. No model inference or training was started."
