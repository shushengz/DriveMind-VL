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
BEST_JSON="${BEST_JSON:-outputs/eval_results/${PREFIX_BASE}_best_selection.json}"
BEST_MD="${BEST_MD:-docs/${PREFIX_BASE}_best_selection.md}"
BEST_MIN_NORMAL_F1="${BEST_MIN_NORMAL_F1:-0.35}"
BEST_MIN_PER_CASE_GAP="${BEST_MIN_PER_CASE_GAP:--0.01}"
CHECKPOINTS="${CHECKPOINTS:-}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-0}"
REUSE_EXISTING="${REUSE_EXISTING:-0}"

mkdir -p outputs/eval_results outputs/logs
: > "${SWEEP_JSONL}"

if [[ -n "${CHECKPOINTS}" ]]; then
  IFS=',' read -r -a CHECKPOINT_LIST <<< "${CHECKPOINTS}"
  for idx in "${!CHECKPOINT_LIST[@]}"; do
    item="${CHECKPOINT_LIST[$idx]}"
    if [[ "${item}" != /* && "${item}" != "${RUN_DIR}" && "${item}" != outputs/* ]]; then
      item="${RUN_DIR}/${item}"
    fi
    CHECKPOINT_LIST[$idx]="${item}"
  done
else
  mapfile -t CHECKPOINT_LIST < <(find "${RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-step-*' | sort)
  CHECKPOINT_LIST+=("${RUN_DIR}")
fi

if [[ "${MAX_CHECKPOINTS}" -gt 0 ]]; then
  CHECKPOINT_LIST=("${CHECKPOINT_LIST[@]:0:${MAX_CHECKPOINTS}}")
fi

for adapter in "${CHECKPOINT_LIST[@]}"; do
  name="$(basename "${adapter}")"
  if [[ "${adapter}" == "${RUN_DIR}" ]]; then
    name="final"
  fi
  prefix="${PREFIX_BASE}_${name}"
  echo "== Evaluating ${adapter} as ${prefix} =="
  if [[ "${REUSE_EXISTING}" == "1" \
    && -f "outputs/eval_results/${prefix}_visual_control_summary.json" \
    && -f "outputs/eval_results/${prefix}_visual_control_case_summary.json" ]]; then
    echo "reuse existing visual-control outputs for ${prefix}"
  else
    ADAPTER="${adapter}" \
    SPLIT="${SPLIT}" \
    PREFIX="${prefix}" \
    bash scripts/46_eval_lingoqa_clean_visual_controls.sh "${MODEL_DIR}" \
      2>&1 | tee "outputs/logs/${prefix}_eval.log"
  fi

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

python - "$SWEEP_JSONL" "$SWEEP_CSV" "$BEST_JSON" "$BEST_MD" "$BEST_MIN_NORMAL_F1" "$BEST_MIN_PER_CASE_GAP" <<'PY'
import csv
import json
import shutil
import sys
from pathlib import Path

jsonl_path, csv_path, best_json, best_md = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
min_normal_f1, min_per_case_gap = float(sys.argv[5]), float(sys.argv[6])
rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if rows:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def is_candidate(row):
        return (
            float(row.get("normal_f1", 0.0)) >= min_normal_f1
            and float(row.get("per_case_gap", 0.0)) >= min_per_case_gap
            and float(row.get("setting_gap", 0.0)) > 0.0
        )

    candidates = [row for row in rows if is_candidate(row)]
    if candidates:
        selected = sorted(
            candidates,
            key=lambda row: (
                float(row.get("normal_f1", 0.0)),
                float(row.get("per_case_gap", 0.0)),
                float(row.get("setting_gap", 0.0)),
            ),
            reverse=True,
        )[0]
        status = "candidate"
    else:
        selected = sorted(
            rows,
            key=lambda row: (
                float(row.get("normal_f1", 0.0)),
                float(row.get("per_case_gap", 0.0)),
                float(row.get("setting_gap", 0.0)),
            ),
            reverse=True,
        )[0]
        status = "fallback_best_available"

    prefix = str(selected["prefix"])
    summary_src = Path(f"outputs/eval_results/{prefix}_visual_control_summary.json")
    case_src = Path(f"outputs/eval_results/{prefix}_visual_control_case_summary.json")
    summary_dst = Path(f"outputs/eval_results/{jsonl_path.stem.replace('_summary', '')}_best_visual_control_summary.json")
    case_dst = Path(f"outputs/eval_results/{jsonl_path.stem.replace('_summary', '')}_best_visual_control_case_summary.json")
    if summary_src.exists():
        shutil.copyfile(summary_src, summary_dst)
    if case_src.exists():
        shutil.copyfile(case_src, case_dst)

    selection = {
        "status": status,
        "thresholds": {
            "min_normal_f1": min_normal_f1,
            "min_per_case_gap": min_per_case_gap,
        },
        "selected": selected,
        "selected_summary": summary_dst.as_posix(),
        "selected_case_summary": case_dst.as_posix(),
        "rows": rows,
    }
    best_json.parent.mkdir(parents=True, exist_ok=True)
    best_json.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_md.parent.mkdir(parents=True, exist_ok=True)
    best_md.write_text(
        "\n".join(
            [
                "# Preference-v5 Best Checkpoint Selection",
                "",
                f"Status: `{status}`",
                f"Selected prefix: `{selected['prefix']}`",
                f"Adapter: `{selected['adapter']}`",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| normal_f1 | {float(selected.get('normal_f1', 0.0)):.4f} |",
                f"| text_only_f1 | {float(selected.get('text_only_f1', 0.0)):.4f} |",
                f"| wrong_image_f1 | {float(selected.get('wrong_image_f1', 0.0)):.4f} |",
                f"| blank_image_f1 | {float(selected.get('blank_image_f1', 0.0)):.4f} |",
                f"| setting_gap | {float(selected.get('setting_gap', 0.0)):.4f} |",
                f"| per_case_gap | {float(selected.get('per_case_gap', 0.0)):.4f} |",
                f"| positive_gap_rate | {float(selected.get('positive_gap_rate', 0.0)):.4f} |",
                "",
                f"Copied summary to `{summary_dst}`.",
                f"Copied case summary to `{case_dst}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
print(f"wrote sweep summary to {jsonl_path} and {csv_path}")
if rows:
    print(f"wrote best selection to {best_json} and {best_md}")
PY
