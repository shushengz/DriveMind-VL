#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS_OVERRIDE:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS_OVERRIDE:-4}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct}"
RUN_DIR="${RUN_DIR:-outputs/checkpoints/qwen25vl_3b_lingoqa_pref_v5_from_predictions}"
SPLIT="${SPLIT:-dev}"
PREFIX_BASE="${PREFIX_BASE:-lingoqa_clean_v2_${SPLIT}_pref_v5_sweep}"
SWEEP_JSONL="${SWEEP_JSONL:-outputs/eval_results/${PREFIX_BASE}_summary.jsonl}"
SWEEP_CSV="${SWEEP_CSV:-outputs/eval_results/${PREFIX_BASE}_summary.csv}"

mkdir -p outputs/eval_results outputs/logs
: > "${SWEEP_JSONL}"

mapfile -t CHECKPOINTS < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-step-*' | sort)
CHECKPOINTS+=("${RUN_DIR}")

for adapter in "${CHECKPOINTS[@]}"; do
  name="$(basename "${adapter}")"
  if [[ "${adapter}" == "${RUN_DIR}" ]]; then
    name="final"
  fi
  prefix="${PREFIX_BASE}_${name}"
  echo "== Evaluating ${adapter} as ${prefix} =="
  ADAPTER="${adapter}" \
  SPLIT="${SPLIT}" \
  PREFIX="${prefix}" \
  bash scripts/46_eval_lingoqa_clean_visual_controls.sh "${MODEL_DIR}" \
    2>&1 | tee "outputs/logs/${prefix}_eval.log"

  python - "$prefix" "$adapter" "$SWEEP_JSONL" <<'PY'
import json
import sys
from pathlib import Path

prefix, adapter, out_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
summary_path = Path(f"outputs/eval_results/{prefix}_visual_control_summary.json")
case_path = Path(f"outputs/eval_results/{prefix}_visual_control_case_summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
case_summary = json.loads(case_path.read_text(encoding="utf-8"))
dep = summary.get("_visual_dependency", {})
overall = case_summary.get("overall", {})
row = {
    "prefix": prefix,
    "adapter": adapter,
    "normal_f1": summary.get("normal", {}).get("external_answer_f1", 0.0),
    "text_only_f1": summary.get("text_only", {}).get("external_answer_f1", 0.0),
    "wrong_image_f1": summary.get("wrong_image", {}).get("external_answer_f1", 0.0),
    "blank_image_f1": summary.get("blank_image", {}).get("external_answer_f1", 0.0),
    "setting_gap": dep.get("visual_dependency_gap", 0.0),
    "per_case_gap": overall.get("visual_dependency_gap", 0.0),
    "positive_gap_rate": overall.get("positive_gap_rate", 0.0),
    "normal_refusal_rate": dep.get("normal_refusal_rate", 0.0),
    "control_refusal_rate": dep.get("control_refusal_rate", 0.0),
}
with out_path.open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False, indent=2))
PY
done

python - "$SWEEP_JSONL" "$SWEEP_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl_path, csv_path = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if rows:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(f"wrote sweep summary to {jsonl_path} and {csv_path}")
PY
