#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
export CUDA_VISIBLE_DEVICES=""

RUN=0
DO_DIS=0; DO_MANUAL=0; DO_DEBUG=0; DO_SENS=0; DO_PAIR=0; DO_COMPARE=0; DO_DIAG=0; DO_DOCS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --collect_disagreements) DO_DIS=1; shift ;;
    --manual_review) DO_MANUAL=1; shift ;;
    --debug_v2_1) DO_DEBUG=1; shift ;;
    --sensitivity) DO_SENS=1; shift ;;
    --pairwise) DO_PAIR=1; shift ;;
    --compare) DO_COMPARE=1; shift ;;
    --diagnose) DO_DIAG=1; shift ;;
    --update_docs) DO_DOCS=1; shift ;;
    --all) DO_DIS=1; DO_MANUAL=1; DO_DEBUG=1; DO_SENS=1; DO_PAIR=1; DO_COMPARE=1; DO_DIAG=1; DO_DOCS=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "[stage14.5] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_DIS$DO_MANUAL$DO_DEBUG$DO_SENS$DO_PAIR$DO_COMPARE$DO_DIAG$DO_DOCS" == "00000000" ]]; then
  echo "Usage: bash scripts/run_stage14_5_reward_v2_1_cpu.sh --all --run"
  exit 0
fi
echo "Stage 14.5: reward v2.1 calibration and disagreement review only. CUDA disabled; no model loading, inference, training, downloads, or GRPO."
commands=()
[[ "$DO_DIS" == "1" ]] && commands+=("$PYTHON_BIN src/rl/collect_reward_v2_disagreement_cases.py")
[[ "$DO_MANUAL" == "1" ]] && commands+=("$PYTHON_BIN src/rl/build_reward_manual_review_sheet.py")
[[ "$DO_DEBUG" == "1" ]] && commands+=("$PYTHON_BIN src/rl/debug_reward_vc_grpo_lite_v2_1.py")
[[ "$DO_SENS" == "1" ]] && commands+=("$PYTHON_BIN src/rl/analyze_reward_v2_1_sensitivity.py")
[[ "$DO_PAIR" == "1" ]] && commands+=("$PYTHON_BIN src/rl/evaluate_reward_v2_1_pairwise.py")
[[ "$DO_COMPARE" == "1" ]] && commands+=("$PYTHON_BIN src/rl/compare_reward_v2_vs_v2_1.py")
[[ "$DO_DIAG" == "1" ]] && commands+=("$PYTHON_BIN src/rl/diagnose_reward_v2_1_readiness.py")
[[ "$DO_DOCS" == "1" ]] && commands+=("$PYTHON_BIN src/rl/update_stage14_5_reward_v2_1_docs.py")
if [[ "$RUN" != "1" ]]; then
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi
if [[ "$DO_DEBUG" == "1" || "$DO_PAIR" == "1" ]]; then
  "$PYTHON_BIN" -m pytest -q tests/test_reward_vc_grpo_lite_v2_1.py
fi
for command in "${commands[@]}"; do
  echo "[stage14.5] $command"
  eval "$command"
done
echo "[stage14.5] complete. CUDA remained disabled and no inference/training command was called."
