#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DO_COLLECT=0
DO_BUILD=0
DO_AUDIT=0
DO_CONFIG=0
DO_DIAGNOSE=0
RUN=0
DRY_RUN=1
MAX_PAIRS=500
SEED=42

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect) DO_COLLECT=1 ;;
    --build_pairs) DO_BUILD=1 ;;
    --audit) DO_AUDIT=1 ;;
    --make_config) DO_CONFIG=1 ;;
    --diagnose) DO_DIAGNOSE=1 ;;
    --all) DO_COLLECT=1; DO_BUILD=1; DO_AUDIT=1; DO_CONFIG=1; DO_DIAGNOSE=1 ;;
    --run) RUN=1; DRY_RUN=0 ;;
    --dry_run) DRY_RUN=1; RUN=0 ;;
    --max_pairs) MAX_PAIRS="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    *) echo "[stage9] unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$DO_COLLECT$DO_BUILD$DO_AUDIT$DO_CONFIG$DO_DIAGNOSE" == "00000" ]]; then
  DO_COLLECT=1; DO_BUILD=1; DO_AUDIT=1; DO_CONFIG=1; DO_DIAGNOSE=1
fi

echo "CPU-only Stage 9: no model training, no model inference, no GPU usage."
echo "[stage9] mode=$([[ "$DRY_RUN" -eq 1 ]] && echo dry_run || echo run) max_pairs=$MAX_PAIRS seed=$SEED"

if [[ "$DRY_RUN" -eq 1 ]]; then
  MINING_OUT="outputs/stage9_dry_run/data/mining/v8_2"
  TRAIN_OUT="outputs/stage9_dry_run/data/train/preference_v8_2"
  AUDIT_OUT="outputs/stage9_dry_run/data_audit"
  FINAL_OUT="outputs/stage9_dry_run/final_report"
  EXTRA=(--dry_run)
  DIAG_EXTRA=(--dry_run)
else
  MINING_OUT="data/mining/v8_2"
  TRAIN_OUT="data/train/preference_v8_2"
  AUDIT_OUT="outputs/data_audit"
  FINAL_OUT="outputs/final_report"
  EXTRA=(--run)
  DIAG_EXTRA=()
fi
mkdir -p "$FINAL_OUT" "$AUDIT_OUT"

if [[ "$DO_COLLECT" -eq 1 ]]; then
  echo "[stage9] 1/5 collect candidate cases"
  "$PYTHON_BIN" src/data/collect_preference_v8_2_cases.py \
    --output_dir "$MINING_OUT" \
    --report "$AUDIT_OUT/preference_v8_2_candidate_report.md" \
    --seed "$SEED" "${EXTRA[@]}"
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "[stage9] 2/5 build Preference-v8.2 pairs"
  "$PYTHON_BIN" src/data/build_preference_v8_2.py \
    --candidates "$MINING_OUT/preference_v8_2_candidate_cases.jsonl" \
    --output_dir "$TRAIN_OUT" \
    --max_pairs "$MAX_PAIRS" \
    --seed "$SEED" "${EXTRA[@]}"
fi

if [[ "$DO_AUDIT" -eq 1 ]]; then
  echo "[stage9] 3/5 audit Preference-v8.2"
  "$PYTHON_BIN" src/data/audit_preference_v8_2.py \
    --preference_file "$TRAIN_OUT/preference_v8_2_pairs.jsonl" \
    --candidates "$MINING_OUT/preference_v8_2_candidate_cases.jsonl" \
    --output_dir "$AUDIT_OUT" "${EXTRA[@]}"
fi

if [[ "$DO_CONFIG" -eq 1 ]]; then
  echo "[stage9] 4/5 verify DPO-v8.2 smoke config draft"
  CONFIG_OUT="configs/dpo_v8_2_lingo_smoke.yaml"
  test -f "$CONFIG_OUT"
  grep -q '^init_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/' "$CONFIG_OUT"
  grep -q '^reference_adapter: checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/' "$CONFIG_OUT"
  grep -q '^reference_free: false' "$CONFIG_OUT"
  grep -q '^learning_rate: 5.0e-8' "$CONFIG_OUT"
  grep -q '^max_steps: 25' "$CONFIG_OUT"
fi

if [[ "$DO_DIAGNOSE" -eq 1 ]]; then
  echo "[stage9] 5/5 diagnose readiness"
  "$PYTHON_BIN" src/eval/diagnose_stage9_preference_v8_2.py \
    --stats "$TRAIN_OUT/preference_v8_2_stats.json" \
    --audit "$AUDIT_OUT/preference_v8_2_audit.json" \
    --output_md "$FINAL_OUT/stage9_preference_v8_2_diagnosis.md" \
    --output_json "$FINAL_OUT/stage9_preference_v8_2_diagnosis.json" "${DIAG_EXTRA[@]}"
fi

echo "[stage9] complete; no training or inference was invoked."
