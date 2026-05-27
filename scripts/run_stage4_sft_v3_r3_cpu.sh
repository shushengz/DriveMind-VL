#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${OMP_NUM_THREADS:-}" || ! "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ || "${OMP_NUM_THREADS:-0}" -eq 0 ]]; then export OMP_NUM_THREADS=1; fi

echo "CPU-only Stage 4: no model training, no model inference, no GPU usage."
RUN=0; DO_BUILD=0; DO_AUDIT=0; DO_RESCORE=0; DO_MAIN=0; DO_DIAG=0; DO_GALLERY=0; DO_TEST=0; DO_ALL=0
MAX_SAMPLES=0; SEED=42; PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;; --dry_run) RUN=0; shift ;;
    --build_data) DO_BUILD=1; shift ;; --audit) DO_AUDIT=1; shift ;; --rescore) DO_RESCORE=1; shift ;; --main_results) DO_MAIN=1; shift ;; --diagnose) DO_DIAG=1; shift ;; --case_gallery) DO_GALLERY=1; shift ;; --test) DO_TEST=1; shift ;; --all) DO_ALL=1; shift ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;; --seed) SEED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_ALL" -eq 1 ]]; then DO_BUILD=1; DO_AUDIT=1; DO_RESCORE=1; DO_MAIN=1; DO_DIAG=1; DO_GALLERY=1; DO_TEST=1; fi
if [[ "$DO_BUILD$DO_AUDIT$DO_RESCORE$DO_MAIN$DO_DIAG$DO_GALLERY$DO_TEST" == "0000000" ]]; then DO_ALL=1; DO_BUILD=1; DO_AUDIT=1; DO_RESCORE=1; DO_MAIN=1; DO_DIAG=1; DO_GALLERY=1; DO_TEST=1; fi
cd "$(dirname "$0")/.."
DRY_FLAG="--dry_run"; ROOT_OUT="outputs/stage4_dry_run"; DATA_OUT="$ROOT_OUT/data/train/sft_v3_r3"
if [[ "$RUN" -eq 1 ]]; then DRY_FLAG=""; ROOT_OUT="outputs"; DATA_OUT="data/train/sft_v3_r3"; fi
MAX_FLAG=""; if [[ "$MAX_SAMPLES" != "0" ]]; then MAX_FLAG="--max_samples $MAX_SAMPLES"; fi
mkdir -p "$ROOT_OUT/final_report" "$ROOT_OUT/data_audit"
echo "[stage4] mode=$([[ $RUN -eq 1 ]] && echo run || echo dry_run) seed=$SEED"
if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "[stage4] build r3 data"
  "$PYTHON_BIN" src/data/build_sft_v3_r3.py $DRY_FLAG --output_dir "$DATA_OUT" --seed "$SEED" $MAX_FLAG
fi
if [[ "$DO_AUDIT" -eq 1 ]]; then
  echo "[stage4] audit r3 data"
  "$PYTHON_BIN" src/data/audit_sft_v3_r3.py --input "$DATA_OUT/lingoqa_sft_v3_r3.jsonl" --output_json "$ROOT_OUT/data_audit/sft_v3_r3_lingo_audit.json" --output_md "$ROOT_OUT/data_audit/sft_v3_r3_lingo_audit.md" --seed "$SEED"
fi
if [[ "$DO_RESCORE" -eq 1 ]]; then
  echo "[stage4] answer-only rescore existing models"
  "$PYTHON_BIN" src/eval/rescore_answer_only.py $DRY_FLAG --dataset lingoqa --allow_missing --output "$ROOT_OUT/final_report/stage4_answer_only_rescore.csv"
fi
if [[ "$DO_MAIN" -eq 1 ]]; then
  echo "[stage4] build main results"
  "$PYTHON_BIN" src/eval/build_stage4_main_results.py $DRY_FLAG --dataset lingoqa --output "$ROOT_OUT/final_report/stage4_sft_v3_r3_main_results.csv" --warnings_output "$ROOT_OUT/final_report/stage4_sft_v3_r3_main_results_warnings.json"
fi
if [[ "$DO_DIAG" -eq 1 ]]; then
  echo "[stage4] diagnose"
  "$PYTHON_BIN" src/eval/diagnose_stage4_sft_v3_r3.py --input "$ROOT_OUT/final_report/stage4_sft_v3_r3_main_results.csv" --output_json "$ROOT_OUT/final_report/stage4_sft_v3_r3_diagnosis.json" --output_md "$ROOT_OUT/final_report/stage4_sft_v3_r3_diagnosis.md"
fi
if [[ "$DO_GALLERY" -eq 1 ]]; then
  echo "[stage4] case gallery"
  "$PYTHON_BIN" src/eval/build_stage4_sft_v3_r3_case_gallery.py --output_csv "$ROOT_OUT/final_report/stage4_sft_v3_r3_case_gallery.csv" --output_html "$ROOT_OUT/final_report/stage4_sft_v3_r3_case_gallery.html"
fi
if [[ "$DO_TEST" -eq 1 ]]; then
  echo "[stage4] pytest"
  "$PYTHON_BIN" -m pytest tests/test_prediction_parser.py tests/test_rescore_answer_only.py tests/test_build_sft_v3_r3.py tests/test_audit_sft_v3_r3.py
fi
echo "[stage4] done"
