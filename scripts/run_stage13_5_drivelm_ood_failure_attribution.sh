#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
export CUDA_VISIBLE_DEVICES=""

RUN=0
DO_ALIGNMENT=0; DO_TAXONOMY=0; DO_SPATIAL=0; DO_FIX_BREAK=0; DO_CONTROL=0
DO_WRITEUP=0; DO_REWARD=0; DO_REPORT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --alignment) DO_ALIGNMENT=1; shift ;;
    --taxonomy) DO_TAXONOMY=1; shift ;;
    --spatial) DO_SPATIAL=1; shift ;;
    --fix_break) DO_FIX_BREAK=1; shift ;;
    --control_behavior) DO_CONTROL=1; shift ;;
    --writeup) DO_WRITEUP=1; shift ;;
    --reward_implications) DO_REWARD=1; shift ;;
    --update_report) DO_REPORT=1; shift ;;
    --all) DO_ALIGNMENT=1; DO_TAXONOMY=1; DO_SPATIAL=1; DO_FIX_BREAK=1; DO_CONTROL=1; DO_WRITEUP=1; DO_REWARD=1; DO_REPORT=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "[stage13.5] unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ "$DO_ALIGNMENT$DO_TAXONOMY$DO_SPATIAL$DO_FIX_BREAK$DO_CONTROL$DO_WRITEUP$DO_REWARD$DO_REPORT" == "00000000" ]]; then
  echo "Usage: bash scripts/run_stage13_5_drivelm_ood_failure_attribution.sh --all --run"
  exit 0
fi
echo "Stage 13.5: offline DriveLM OOD failure attribution only. CUDA disabled; no model loading, inference, training, or GRPO."
commands=()
[[ "$DO_ALIGNMENT" == "1" ]] && commands+=("$PYTHON_BIN src/eval/analyze_drivelm_ood_alignment.py")
[[ "$DO_TAXONOMY" == "1" ]] && commands+=("$PYTHON_BIN src/eval/analyze_drivelm_ood_failure_taxonomy.py")
[[ "$DO_SPATIAL" == "1" ]] && commands+=("$PYTHON_BIN src/eval/analyze_drivelm_spatial_failures.py")
[[ "$DO_FIX_BREAK" == "1" ]] && commands+=("$PYTHON_BIN src/eval/analyze_drivelm_base_vs_r3_fix_break.py")
[[ "$DO_CONTROL" == "1" ]] && commands+=("$PYTHON_BIN src/eval/analyze_drivelm_control_behavior.py")
[[ "$DO_WRITEUP" == "1" ]] && commands+=("$PYTHON_BIN src/eval/write_stage13_5_drivelm_documents.py --writeup")
[[ "$DO_REWARD" == "1" ]] && commands+=("$PYTHON_BIN src/eval/write_stage13_5_drivelm_documents.py --reward_implications")
[[ "$DO_REPORT" == "1" ]] && commands+=("$PYTHON_BIN src/eval/write_stage13_5_drivelm_documents.py --update_report")
if [[ "$RUN" != "1" ]]; then
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi
for command in "${commands[@]}"; do
  echo "[stage13.5] $command"
  eval "$command"
done
echo "[stage13.5] complete. CUDA remained disabled and no inference/training command was called."
