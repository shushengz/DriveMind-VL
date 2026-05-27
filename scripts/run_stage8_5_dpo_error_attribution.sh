#!/usr/bin/env bash
set -euo pipefail

echo "CPU-only Stage 8.5: no model training, no model inference, no GPU usage."
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
RUN=0
DO_BEHAVIOR=0; DO_GAP=0; DO_FIX_BREAK=0; DO_DESIGN=0; DO_DIAGNOSE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --behavior) DO_BEHAVIOR=1; shift ;;
    --gap_regression) DO_GAP=1; shift ;;
    --fix_break) DO_FIX_BREAK=1; shift ;;
    --design_v8_2) DO_DESIGN=1; shift ;;
    --diagnose) DO_DIAGNOSE=1; shift ;;
    --all) DO_BEHAVIOR=1; DO_GAP=1; DO_FIX_BREAK=1; DO_DESIGN=1; DO_DIAGNOSE=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "[stage8.5] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_BEHAVIOR$DO_GAP$DO_FIX_BREAK$DO_DESIGN$DO_DIAGNOSE" == "00000" ]]; then
  DO_BEHAVIOR=1; DO_GAP=1; DO_FIX_BREAK=1; DO_DESIGN=1; DO_DIAGNOSE=1
fi
if [[ "$RUN" == "1" ]]; then
  OUT="outputs/final_report"; CONFIG="configs/dpo_v8_2_lingo_smoke.yaml"; DRY_ARGS=()
else
  OUT="outputs/stage8_5_dry_run/final_report"; CONFIG="outputs/stage8_5_dry_run/configs/dpo_v8_2_lingo_smoke.yaml"; DRY_ARGS=(--dry_run)
fi
mkdir -p "$OUT"
echo "[stage8.5] output_dir=${OUT}"
if [[ "$DO_BEHAVIOR" == "1" ]]; then
  echo "[stage8.5] 1/5 control behavior metrics"
  "$PYTHON_BIN" src/eval/analyze_stage8_control_behavior.py --output_dir "$OUT" "${DRY_ARGS[@]}"
fi
if [[ "$DO_GAP" == "1" ]]; then
  echo "[stage8.5] 2/5 case-gap regression"
  "$PYTHON_BIN" src/eval/analyze_case_gap_regression.py --output_csv "$OUT/stage8_5_case_gap_regression.csv" --output_md "$OUT/stage8_5_case_gap_regression.md" "${DRY_ARGS[@]}"
fi
if [[ "$DO_FIX_BREAK" == "1" ]]; then
  echo "[stage8.5] 3/5 DPO fix/break analysis"
  "$PYTHON_BIN" src/eval/analyze_dpo_fix_break_cases.py --output_fix "$OUT/stage8_5_dpo_fix_cases.csv" --output_break "$OUT/stage8_5_dpo_break_cases.csv" --output_md "$OUT/stage8_5_dpo_fix_break_report.md" "${DRY_ARGS[@]}"
fi
if [[ "$DO_DESIGN" == "1" ]]; then
  echo "[stage8.5] 4/5 Preference-v8.2 design proposal"
  "$PYTHON_BIN" src/data/design_preference_v8_2.py --behavior_by_setting "$OUT/stage8_5_control_behavior_by_setting.csv" --regression "$OUT/stage8_5_case_gap_regression.csv" --fix_cases "$OUT/stage8_5_dpo_fix_cases.csv" --output_md "$OUT/preference_v8_2_design.md" --output_json "$OUT/preference_v8_2_design.json" --config_output "$CONFIG" "${DRY_ARGS[@]}"
fi
if [[ "$DO_DIAGNOSE" == "1" ]]; then
  echo "[stage8.5] 5/5 final attribution diagnosis"
  "$PYTHON_BIN" src/eval/diagnose_stage8_5_dpo_error_attribution.py --summary "$OUT/stage8_5_control_behavior_summary.csv" --by_setting "$OUT/stage8_5_control_behavior_by_setting.csv" --regression "$OUT/stage8_5_case_gap_regression.csv" --fix_cases "$OUT/stage8_5_dpo_fix_cases.csv" --break_cases "$OUT/stage8_5_dpo_break_cases.csv" --output_md "$OUT/stage8_5_error_attribution.md" --output_json "$OUT/stage8_5_error_attribution.json" "${DRY_ARGS[@]}"
fi
echo "[stage8.5] README_CN.md documents this offline workflow. No training or inference was launched."
