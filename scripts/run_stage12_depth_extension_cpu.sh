#!/usr/bin/env bash
set -euo pipefail

echo "Stage 12 Depth Extension Preparation: CPU-only data/reward/report workflow. No training, inference, model loading, or downloads."
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
SEED=42; RUN=0
DO_LINGO=0; DO_DRIVE=0; DO_REWARD=0; DO_BLUEPRINT=0; DO_ROADMAP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lingoqa_pool) DO_LINGO=1; shift ;;
    --drivelm_pool) DO_DRIVE=1; shift ;;
    --reward_debug) DO_REWARD=1; shift ;;
    --grpo_blueprint) DO_BLUEPRINT=1; shift ;;
    --roadmap) DO_ROADMAP=1; shift ;;
    --all) DO_LINGO=1; DO_DRIVE=1; DO_REWARD=1; DO_BLUEPRINT=1; DO_ROADMAP=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "[stage12] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_LINGO$DO_DRIVE$DO_REWARD$DO_BLUEPRINT$DO_ROADMAP" == "00000" ]]; then
  DO_LINGO=1; DO_DRIVE=1; DO_REWARD=1; DO_BLUEPRINT=1; DO_ROADMAP=1
fi
if [[ "$RUN" != "1" ]]; then
  echo "[stage12] dry-run plan seed=${SEED}:"
  [[ "$DO_LINGO" == "1" ]] && echo "- build leakage-filtered LingoQA held-out ID files and merged pool"
  [[ "$DO_DRIVE" == "1" ]] && echo "- validate and build DriveLM OOD strict visual-control pool"
  [[ "$DO_REWARD" == "1" ]] && echo "- score existing held-out predictions with offline reward harness"
  [[ "$DO_BLUEPRINT" == "1" ]] && echo "- write GRPO-lite training blueprint only"
  [[ "$DO_ROADMAP" == "1" ]] && echo "- write depth extension roadmap and update README_CN"
  echo "[stage12] future GPU eval scripts are prepared but never invoked by this workflow."
  exit 0
fi

echo "[stage12] run unit tests for offline builders and reward harness"
"$PYTHON_BIN" -m pytest -q tests/test_build_lingoqa_large_heldout_pool.py tests/test_build_drivelm_ood_eval_pool.py tests/test_reward_vc_grpo_lite.py
[[ "$DO_LINGO" == "1" ]] && "$PYTHON_BIN" src/eval/build_lingoqa_large_heldout_pool.py --seed "$SEED"
[[ "$DO_DRIVE" == "1" ]] && "$PYTHON_BIN" src/eval/build_drivelm_ood_eval_pool.py --seed "$SEED"
[[ "$DO_REWARD" == "1" ]] && "$PYTHON_BIN" src/rl/debug_reward_vc_grpo_lite.py
if [[ "$DO_BLUEPRINT" == "1" || "$DO_ROADMAP" == "1" ]]; then
  [[ -f outputs/final_report/lingoqa_large_heldout_pool_summary.json && -f outputs/final_report/drivelm_ood_pool_summary.json && -f outputs/final_report/grpo_lite_reward_summary.csv ]] || { echo "summaries missing; run --lingoqa_pool --drivelm_pool --reward_debug first" >&2; exit 1; }
  "$PYTHON_BIN" src/eval/write_stage12_depth_extension_docs.py
fi
echo "[stage12] complete. CUDA remained disabled and no inference/training command was called."
