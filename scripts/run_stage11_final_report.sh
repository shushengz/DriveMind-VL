#!/usr/bin/env bash
set -euo pipefail

echo "Stage 11 final consolidation: offline artifacts only; no training, inference, or GPU use."
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
RUN=0
DO_COLLECT=0; DO_SELECT=0; DO_ABLATION=0; DO_GALLERY=0; DO_README=0; DO_RESUME=0; DO_GRPO=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect) DO_COLLECT=1; shift ;;
    --select) DO_SELECT=1; shift ;;
    --ablation) DO_ABLATION=1; shift ;;
    --gallery) DO_GALLERY=1; shift ;;
    --readme_results) DO_README=1; shift ;;
    --resume) DO_RESUME=1; shift ;;
    --grpo_plan) DO_GRPO=1; shift ;;
    --all) DO_COLLECT=1; DO_SELECT=1; DO_ABLATION=1; DO_GALLERY=1; DO_README=1; DO_RESUME=1; DO_GRPO=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "[stage11] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_COLLECT$DO_SELECT$DO_ABLATION$DO_GALLERY$DO_README$DO_RESUME$DO_GRPO" == "0000000" ]]; then
  DO_COLLECT=1; DO_SELECT=1; DO_ABLATION=1; DO_GALLERY=1; DO_README=1; DO_RESUME=1; DO_GRPO=1
fi
MODE="dry_run"; [[ "$RUN" == "1" ]] && MODE="run"
echo "[stage11] mode=${MODE} CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"

if [[ "$DO_COLLECT" == "1" ]]; then
  echo "[stage11] 1/8 collect final held-out results"
  if [[ "$RUN" == "1" ]]; then
    "$PYTHON_BIN" src/eval/collect_final_experiment_results.py
  else
    "$PYTHON_BIN" src/eval/collect_final_experiment_results.py --dry_run
  fi
fi

if [[ "$RUN" != "1" ]]; then
  echo "[stage11] dry-run complete: downstream writes are deferred. This workflow contains no model-loading command."
  exit 0
fi

if [[ "$DO_SELECT" == "1" ]]; then
  echo "[stage11] 2/8 select final checkpoint"
  "$PYTHON_BIN" src/eval/select_final_checkpoint.py
fi
if [[ "$DO_ABLATION" == "1" ]]; then
  echo "[stage11] 3/8 build ablation tables"
  "$PYTHON_BIN" src/eval/build_final_ablation_tables.py
fi
if [[ "$DO_GALLERY" == "1" ]]; then
  echo "[stage11] 4/8 build final case gallery"
  "$PYTHON_BIN" src/eval/build_final_case_gallery.py
fi
if [[ "$DO_README" == "1" ]]; then
  echo "[stage11] 5/8 write results report and consolidate README_CN"
  "$PYTHON_BIN" src/eval/write_stage11_documents.py --readme_results --update_readme_cn
fi
if [[ "$DO_RESUME" == "1" ]]; then
  echo "[stage11] 6/8 generate resume and interview material"
  "$PYTHON_BIN" src/eval/write_stage11_documents.py --resume
fi
if [[ "$DO_GRPO" == "1" ]]; then
  echo "[stage11] 7/8 generate GRPO-lite future plan (design only)"
  "$PYTHON_BIN" src/eval/write_stage11_documents.py --grpo_plan
fi
if [[ "$DO_COLLECT$DO_SELECT$DO_ABLATION$DO_GALLERY$DO_README$DO_RESUME$DO_GRPO" == "1111111" ]]; then
  echo "[stage11] 8/8 check final deliverables"
  "$PYTHON_BIN" src/eval/check_final_deliverables.py
fi
echo "[stage11] complete. No SFT/DPO/GRPO training and no model inference executed."
