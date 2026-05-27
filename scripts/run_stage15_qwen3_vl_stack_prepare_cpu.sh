#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drivemind-vl/bin/python}"
export CUDA_VISIBLE_DEVICES=""
RUN=0
DO_SURVEY=0; DO_MATRIX=0; DO_SFT=0; DO_DPO=0; DO_GRPO=0; DO_SERIES=0; DO_EVAL=0; DO_CONFIGS=0; DO_AUDIT=0; DO_DOCS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --survey) DO_SURVEY=1; shift ;;
    --matrix) DO_MATRIX=1; shift ;;
    --export_sft) DO_SFT=1; shift ;;
    --export_dpo) DO_DPO=1; shift ;;
    --export_grpo) DO_GRPO=1; shift ;;
    --export_qwen_vl_series) DO_SERIES=1; shift ;;
    --make_eval_scripts) DO_EVAL=1; shift ;;
    --make_configs) DO_CONFIGS=1; shift ;;
    --audit) DO_AUDIT=1; shift ;;
    --update_docs) DO_DOCS=1; shift ;;
    --all) DO_SURVEY=1; DO_MATRIX=1; DO_SFT=1; DO_DPO=1; DO_GRPO=1; DO_SERIES=1; DO_EVAL=1; DO_CONFIGS=1; DO_AUDIT=1; DO_DOCS=1; shift ;;
    --run) RUN=1; shift ;;
    --dry_run) RUN=0; shift ;;
    *) echo "[stage15] unknown argument: $1" >&2; exit 2 ;;
  esac
done
echo "Stage 15: Qwen3-VL stack preparation only. CUDA disabled; no model download/load, inference, training, or GRPO."
commands=()
[[ "$DO_SURVEY" == "1" ]] && commands+=("test -s docs/qwen3_vl_finetune_survey.md")
[[ "$DO_MATRIX" == "1" ]] && commands+=("test -s outputs/final_report/qwen3_vl_training_stack_matrix.csv && test -s outputs/final_report/qwen3_vl_training_stack_matrix.md")
[[ "$DO_SFT" == "1" ]] && commands+=("$PYTHON_BIN src/data/export_to_ms_swift_vlm.py")
[[ "$DO_DPO" == "1" ]] && commands+=("$PYTHON_BIN src/data/export_to_ms_swift_dpo.py")
[[ "$DO_GRPO" == "1" ]] && commands+=("$PYTHON_BIN src/data/export_to_ms_swift_grpo.py")
[[ "$DO_SERIES" == "1" ]] && commands+=("$PYTHON_BIN src/data/export_to_qwen_vl_series_finetune.py")
[[ "$DO_EVAL" == "1" ]] && commands+=("test -s scripts/run_qwen3_vl_base_eval.sh && bash -n scripts/run_qwen3_vl_base_eval.sh && test -s src/eval/run_qwen3_vl_visual_control_eval.py && test -s src/eval/build_qwen3_vl_base_eval_results.py")
[[ "$DO_CONFIGS" == "1" ]] && commands+=("test -s configs/ms_swift/qwen3_vl_4b_sft_lingoqa_smoke.yaml && test -s configs/ms_swift/qwen3_vl_8b_sft_lingoqa_smoke.yaml && test -s configs/ms_swift/qwen3_vl_4b_grpo_lite_draft.yaml")
[[ "$DO_AUDIT" == "1" ]] && commands+=("$PYTHON_BIN src/data/check_qwen3_vl_exports.py")
[[ "$DO_DOCS" == "1" ]] && commands+=("$PYTHON_BIN src/data/update_stage15_qwen3_vl_docs.py")
if [[ "$RUN" != "1" ]]; then
  printf '[dry-run] %s\n' "${commands[@]}"
  exit 0
fi
"$PYTHON_BIN" -m pytest -q tests/test_drivemind_vl_schema.py
for command in "${commands[@]}"; do
  echo "[stage15] $command"
  eval "$command"
done
echo "[stage15] complete. Only conversion/audit/draft verification ran; CUDA remained disabled."
