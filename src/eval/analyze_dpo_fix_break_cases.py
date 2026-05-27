"""Compare r3 against DPO-v8.1 fixes and regressions using held-out outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics_visual_control import CONTROL_SETTINGS
from src.eval.rescore_answer_only import align_settings, load_model_settings, score_prediction

R3 = "sft_v3_r3_lingo_smoke"
CHECKPOINTS = ("dpo_v8_1_lingo_smoke_step25", "dpo_v8_1_lingo_smoke_step50")
FIELDS = ["checkpoint", "id", "setting", "case_type", "question", "gold", "r3_answer", "dpo_answer", "r3_f1", "dpo_f1", "f1_delta", "r3_hallucination", "dpo_hallucination", "comment"]


def model_map(root: Path, model: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(group["normal"]["id"]): group for group in align_settings(load_model_settings(root, "lingoqa", model, "strict_visual"))}


def score(group: dict[str, dict[str, Any]], setting: str) -> dict[str, Any]:
    return score_prediction(group[setting], "answer_only")


def record(checkpoint: str, sample_id: str, setting: str, kind: str, source: dict[str, Any], r3: dict[str, Any], dpo: dict[str, Any], comment: str) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint, "id": sample_id, "setting": setting, "case_type": kind,
        "question": source["normal"].get("question", ""), "gold": r3["gold"],
        "r3_answer": r3["score_text"], "dpo_answer": dpo["score_text"],
        "r3_f1": r3["f1"], "dpo_f1": dpo["f1"], "f1_delta": dpo["f1"] - r3["f1"],
        "r3_hallucination": r3["hallucination"], "dpo_hallucination": dpo["hallucination"], "comment": comment,
    }


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage 8 DPO fixes and breaks.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--output_fix", default="outputs/final_report/stage8_5_dpo_fix_cases.csv")
    parser.add_argument("--output_break", default="outputs/final_report/stage8_5_dpo_break_cases.csv")
    parser.add_argument("--output_md", default="outputs/final_report/stage8_5_dpo_fix_break_report.md")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(args.prediction_root)
    r3_map = model_map(root, R3)
    fixes, breaks = [], []
    fix_counts, break_counts = Counter(), Counter()
    for checkpoint in CHECKPOINTS:
        current_map = model_map(root, checkpoint)
        ids = sorted(set(r3_map) & set(current_map))
        if args.dry_run:
            ids = ids[:8]
        for sample_id in ids:
            source = r3_map[sample_id]
            r3_normal = score(source, "normal")
            dpo_normal = score(current_map[sample_id], "normal")
            if r3_normal["f1"] >= 0.30 and dpo_normal["f1"] < r3_normal["f1"] - 0.15:
                breaks.append(record(checkpoint, sample_id, "normal", "normal_answer_degraded", source, r3_normal, dpo_normal, "DPO lowered a previously usable normal answer."))
                break_counts[(checkpoint, "normal_answer_degraded")] += 1
            for setting in CONTROL_SETTINGS:
                r3_control = score(source, setting)
                dpo_control = score(current_map[sample_id], setting)
                hallucination_fixed = r3_control["hallucination"] and not dpo_control["hallucination"]
                overlap_fixed = r3_control["f1"] >= 0.30 and dpo_control["f1"] < 0.20
                if hallucination_fixed or overlap_fixed:
                    kind = "control_hallucination_fixed" if hallucination_fixed else "control_high_f1_fixed"
                    text = "DPO removed a control hallucination." if hallucination_fixed else "DPO reduced a high gold-overlap control answer."
                    fixes.append(record(checkpoint, sample_id, setting, kind, source, r3_control, dpo_control, text))
                    fix_counts[(checkpoint, kind)] += 1
                if dpo_control["f1"] > r3_control["f1"] + 0.15:
                    breaks.append(record(checkpoint, sample_id, setting, "control_gold_overlap_increased", source, r3_control, dpo_control, "DPO made a control answer substantially closer to gold."))
                    break_counts[(checkpoint, setting)] += 1
    write(Path(args.output_fix), fixes)
    write(Path(args.output_break), breaks)
    lines = ["# Stage 8.5 DPO-v8.1 Fix / Break Analysis", "", "## Fix Cases", ""]
    for checkpoint in CHECKPOINTS:
        total = sum(value for (name, _), value in fix_counts.items() if name == checkpoint)
        lines.append(f"- `{checkpoint}`: {total} fix records")
        for (name, setting), value in sorted(fix_counts.items()):
            if name == checkpoint:
                lines.append(f"  - `{setting}`: {value}")
    lines += ["", "## Break Cases", ""]
    for checkpoint in CHECKPOINTS:
        total = sum(value for (name, _), value in break_counts.items() if name == checkpoint)
        lines.append(f"- `{checkpoint}`: {total} break records")
        for (name, kind), value in sorted(break_counts.items()):
            if name == checkpoint:
                lines.append(f"  - `{kind}`: {value}")
    lines += ["", "Fix/break conditions are applied to answer-only held-out predictions; no model inference is involved."]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"fix_count": len(fixes), "break_count": len(breaks), "fix_counts": {f"{a}:{b}": n for (a, b), n in fix_counts.items()}, "break_counts": {f"{a}:{b}": n for (a, b), n in break_counts.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
