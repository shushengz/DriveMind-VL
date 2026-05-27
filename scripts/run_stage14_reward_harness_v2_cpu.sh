#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
export CUDA_VISIBLE_DEVICES=""

RUN=0
DO_RECORDS=0; DO_DEBUG=0; DO_SENSITIVITY=0; DO_CASE=0; DO_DIAGNOSE=0; DO_DOCS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build_records) DO_RECORDS=1; shift ;;
    --debug_reward) DO_DEBUG=1; shift ;;
    --sensitivity) DO_SENSITIVITY=1; shift ;;
    --case_study) DO_CASE=1; shift ;;
    --diagnose) DO_DIAGNOSE=1; shift ;;
    --update_docs) DO_DOCS=1; shift ;;
    --all) DO_RECORDS=1; DO_DEBUG=1; DO_SENSITIVITY=1; DO_CASE=1; DO_DIAGNOSE=1; DO_DOCS=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "[stage14] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_RECORDS$DO_DEBUG$DO_SENSITIVITY$DO_CASE$DO_DIAGNOSE$DO_DOCS" == "000000" ]]; then
  echo "Usage: bash scripts/run_stage14_reward_harness_v2_cpu.sh --all --run"
  exit 0
fi
echo "Stage 14: GRPO-lite reward harness v2 offline audit only. CUDA disabled; no model loading, inference, training, or GRPO."
commands=()
[[ "$DO_RECORDS" == "1" ]] && commands+=("$PYTHON_BIN src/rl/build_reward_eval_records.py")
[[ "$DO_DEBUG" == "1" ]] && commands+=("$PYTHON_BIN src/rl/debug_reward_vc_grpo_lite_v2.py")
[[ "$DO_SENSITIVITY" == "1" ]] && commands+=("$PYTHON_BIN src/rl/analyze_reward_v2_sensitivity.py")
[[ "$DO_CASE" == "1" ]] && commands+=("$PYTHON_BIN src/rl/build_reward_v2_case_study.py")
[[ "$DO_DIAGNOSE" == "1" ]] && commands+=("$PYTHON_BIN src/rl/diagnose_reward_v2_readiness.py")
[[ "$DO_DOCS" == "1" ]] && commands+=("$PYTHON_BIN src/rl/update_stage14_reward_v2_docs.py")
if [[ "$RUN" != "1" ]]; then
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi
if [[ "$DO_RECORDS" == "1" || "$DO_DEBUG" == "1" ]]; then
  "$PYTHON_BIN" -m pytest -q tests/test_reward_vc_grpo_lite_v2.py
fi
for command in "${commands[@]}"; do
  echo "[stage14] $command"
  eval "$command"
done
echo "[stage14] complete. CUDA remained disabled and no inference/training command was called."
