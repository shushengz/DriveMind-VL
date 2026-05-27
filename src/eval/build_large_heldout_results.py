"""Aggregate future LingoQA larger held-out predictions using answer-only scores."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.control_behavior_metrics import behavior_flags
from src.eval.metrics_visual_control import CONTROL_SETTINGS, mean
from src.eval.rescore_answer_only import align_settings, load_model_settings, summarize_aligned, score_prediction

MODELS = ["base_qwen25vl_3b", "sft_v2", "sft_v3_r3_lingo_smoke", "dpo_v8_2_lingo_smoke_step25"]
COLUMNS = ["model_name", "dataset", "num_samples", "score_mode", "normal_f1", "case_gap", "positive_gap_rate", "control_hallucination_rate", "control_direct_answer_rate", "control_high_f1_rate_0_20", "blank_high_f1_rate_0_20", "normal_refusal_rate", "avg_answer_length"]


def behavior(aligned: list[dict[str, dict[str, Any]]]) -> dict[str, float]:
    groups = {setting: [] for setting in CONTROL_SETTINGS}
    for group in aligned:
        for setting in CONTROL_SETTINGS:
            score = score_prediction(group[setting], "answer_only")
            groups[setting].append({**score, **behavior_flags(score["score_text"], score["f1"])})
    all_rows = [row for setting in CONTROL_SETTINGS for row in groups[setting]]
    return {
        "control_direct_answer_rate": mean([float(row["is_direct_answer"]) for row in all_rows]),
        "control_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in all_rows]),
        "blank_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in groups["blank_image"]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build future larger held-out results after GPU inference.")
    parser.add_argument("--prediction_root", default="outputs/predictions_large_heldout")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output", default="outputs/final_report/lingoqa_large_heldout_results.csv")
    parser.add_argument("--diagnosis", default="outputs/final_report/lingoqa_large_heldout_diagnosis.md")
    args = parser.parse_args()
    rows, warnings = [], []
    for model in [name for name in args.models.split(",") if name]:
        try:
            aligned = align_settings(load_model_settings(Path(args.prediction_root), "lingoqa", model, "strict_visual"))
        except Exception as exc:
            warnings.append(f"{model}: {exc}")
            continue
        result = summarize_aligned(aligned, model, "lingoqa", "strict_visual", "answer_only", with_ci=False)
        result.update(behavior(aligned))
        rows.append(result)
    if not rows:
        raise FileNotFoundError("No larger-heldout predictions found. Run the future GPU eval script first.")
    with Path(args.output).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in COLUMNS} for row in rows])
    lines = ["# LingoQA Larger Held-out Diagnosis", "", "This table uses answer-only scores from future GPU-generated predictions.", "", "| model | n | normal_f1 | case_gap | control_high_f1 | blank_high_f1 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['model_name']} | {row['num_samples']} | {row['normal_f1']:.4f} | {row['case_gap']:.4f} | {row['control_high_f1_rate_0_20']:.4f} | {row['blank_high_f1_rate_0_20']:.4f} |")
    if warnings:
        lines += ["", "## Warnings", *[f"- {warning}" for warning in warnings]]
    Path(args.diagnosis).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "warnings": warnings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
