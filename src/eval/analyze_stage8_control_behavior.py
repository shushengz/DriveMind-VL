"""Analyze answer-only control behavior in Stage 8 held-out predictions."""
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
from src.eval.rescore_answer_only import align_settings, load_model_settings, score_prediction

MODELS = ["sft_v3_r3_lingo_smoke", "dpo_v8_lingo_smoke", "dpo_v8_1_lingo_smoke_step25", "dpo_v8_1_lingo_smoke_step50"]
METRIC_COLUMNS = [
    "model_name", "setting", "num_samples", "avg_f1", "direct_answer_rate",
    "short_prior_answer_rate", "caution_rate", "action_answer_rate",
    "count_answer_rate", "high_f1_rate_0_20", "high_f1_rate_0_30",
    "hallucination_rate", "refusal_rate", "avg_answer_length",
]
CASE_COLUMNS = [
    "id", "question", "gold", "setting", "model_name", "answer", "f1",
    "is_direct_answer", "is_short_prior_answer", "is_caution",
    "is_action_answer", "is_count_answer", "is_hallucination",
    "case_gap_contribution", "failure_tags",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in columns} for row in rows])


def summarize(rows: list[dict[str, Any]], model: str, setting: str) -> dict[str, Any]:
    return {
        "model_name": model, "setting": setting, "num_samples": len(rows),
        "avg_f1": mean([float(row["f1"]) for row in rows]),
        "direct_answer_rate": mean([float(row["is_direct_answer"]) for row in rows]),
        "short_prior_answer_rate": mean([float(row["is_short_prior_answer"]) for row in rows]),
        "caution_rate": mean([float(row["is_caution"]) for row in rows]),
        "action_answer_rate": mean([float(row["is_action_answer"]) for row in rows]),
        "count_answer_rate": mean([float(row["is_count_answer"]) for row in rows]),
        "high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in rows]),
        "high_f1_rate_0_30": mean([float(row["high_f1_0_30"]) for row in rows]),
        "hallucination_rate": mean([float(row["is_hallucination"]) for row in rows]),
        "refusal_rate": mean([float(row["is_refusal"]) for row in rows]),
        "avg_answer_length": mean([float(row["answer_length"]) for row in rows]),
    }


def collect_model(root: Path, model: str, dry_run: bool = False) -> list[dict[str, Any]]:
    aligned = align_settings(load_model_settings(root, "lingoqa", model, "strict_visual"))
    if dry_run:
        aligned = aligned[:8]
    out = []
    for group in aligned:
        normal = score_prediction(group["normal"], "answer_only")
        for setting in CONTROL_SETTINGS:
            score = score_prediction(group[setting], "answer_only")
            flags = behavior_flags(score["score_text"], score["f1"])
            tags = [key.removeprefix("is_") for key, value in flags.items() if value and key.startswith("is_")]
            if score["hallucination"]:
                tags.append("hallucination")
            if flags["high_f1_0_20"]:
                tags.append("high_f1_0_20")
            out.append({
                "id": score["id"],
                "question": group[setting].get("question", ""),
                "gold": score["gold"],
                "setting": setting,
                "model_name": model,
                "answer": score["score_text"],
                "f1": score["f1"],
                **flags,
                "is_hallucination": score["hallucination"],
                "is_refusal": score["refusal"],
                "answer_length": score["answer_length"],
                "case_gap_contribution": normal["f1"] - score["f1"],
                "failure_tags": ",".join(tags),
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Stage 8 control behavior attribution.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output_dir", default="outputs/final_report")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    case_rows, summary_rows, setting_rows = [], [], []
    for model in models:
        rows = collect_model(Path(args.prediction_root), model, args.dry_run)
        case_rows.extend(rows)
        summary_rows.append(summarize(rows, model, "all_control"))
        for setting in CONTROL_SETTINGS:
            setting_rows.append(summarize([row for row in rows if row["setting"] == setting], model, setting))
    output = Path(args.output_dir)
    write_csv(output / "stage8_5_control_behavior_summary.csv", summary_rows, METRIC_COLUMNS)
    write_csv(output / "stage8_5_control_behavior_by_setting.csv", setting_rows, METRIC_COLUMNS)
    write_csv(output / "stage8_5_control_behavior_by_case.csv", case_rows, CASE_COLUMNS)
    print(json.dumps({"summary_rows": len(summary_rows), "setting_rows": len(setting_rows), "case_rows": len(case_rows), "output_dir": output.as_posix()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
