"""Locate held-out cases whose case gap regressed after DPO-v8.1."""
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

from src.eval.control_behavior_metrics import behavior_flags
from src.eval.metrics_visual_control import CONTROL_SETTINGS
from src.eval.rescore_answer_only import align_settings, load_model_settings, score_prediction

R3 = "sft_v3_r3_lingo_smoke"
STEP25 = "dpo_v8_1_lingo_smoke_step25"
STEP50 = "dpo_v8_1_lingo_smoke_step50"
FIELDS = [
    "id", "question", "gold",
    "r3_normal_answer", "r3_text_answer", "r3_wrong_answer", "r3_blank_answer",
    "step25_normal_answer", "step25_text_answer", "step25_wrong_answer", "step25_blank_answer",
    "step50_normal_answer", "step50_text_answer", "step50_wrong_answer", "step50_blank_answer",
    "r3_case_gap", "step25_case_gap", "step50_case_gap", "delta_gap_25", "delta_gap_50",
    "main_regression_setting", "regression_reason", "suggested_pair_type",
]


def model_map(root: Path, model: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(group["normal"]["id"]): group for group in align_settings(load_model_settings(root, "lingoqa", model, "strict_visual"))}


def scored(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(row, "answer_only") for setting, row in group.items()}


def gap(scores: dict[str, dict[str, Any]]) -> float:
    return scores["normal"]["f1"] - max(scores[s]["f1"] for s in CONTROL_SETTINGS)


def answer(scores: dict[str, dict[str, Any]], setting: str) -> str:
    return str(scores[setting]["score_text"])


def classify(r3: dict[str, dict[str, Any]], current: dict[str, dict[str, Any]], setting: str) -> tuple[str, str]:
    if current["normal"]["f1"] < r3["normal"]["f1"] - 0.10:
        return "normal_f1_decreased", "normal_anchor_gold_vs_model_wrong"
    before, after = behavior_flags(answer(r3, setting), r3[setting]["f1"]), behavior_flags(answer(current, setting), current[setting]["f1"])
    if before["is_caution"] and after["is_direct_answer"]:
        return "caution_to_direct_answer", "control_direct_answer_vs_caution"
    if setting == "blank_image" and current[setting]["f1"] > r3[setting]["f1"]:
        return "blank_high_f1_increased", "blank_high_f1_vs_caution"
    if setting == "text_only" and (current[setting]["f1"] > r3[setting]["f1"] or after["is_short_prior_answer"]):
        return "text_prior_answer_increased", "text_only_gold_overlap_vs_caution"
    if setting == "wrong_image" and current[setting]["f1"] > r3[setting]["f1"]:
        return "wrong_image_confound_increased", "wrong_image_gold_overlap_vs_caution"
    if current[setting]["answer_length"] != r3[setting]["answer_length"]:
        return "answer_length_changed", "control_direct_answer_vs_caution"
    return "other", "control_direct_answer_vs_caution"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Stage 8 case-gap regressions.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--output_csv", default="outputs/final_report/stage8_5_case_gap_regression.csv")
    parser.add_argument("--output_md", default="outputs/final_report/stage8_5_case_gap_regression.md")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(args.prediction_root)
    maps = {model: model_map(root, model) for model in (R3, STEP25, STEP50)}
    ids = sorted(set.intersection(*(set(values) for values in maps.values())))
    if args.dry_run:
        ids = ids[:8]
    rows = []
    for sample_id in ids:
        r3, s25, s50 = (scored(maps[model][sample_id]) for model in (R3, STEP25, STEP50))
        r3_gap, gap25, gap50 = gap(r3), gap(s25), gap(s50)
        delta25, delta50 = gap25 - r3_gap, gap50 - r3_gap
        if min(delta25, delta50) >= 0:
            continue
        focus = s50 if delta50 <= delta25 else s25
        setting = max(CONTROL_SETTINGS, key=lambda name: focus[name]["f1"] - r3[name]["f1"])
        reason, pair_type = classify(r3, focus, setting)
        if reason == "normal_f1_decreased":
            setting = "normal"
        rows.append({
            "id": sample_id,
            "question": maps[R3][sample_id]["normal"].get("question", ""),
            "gold": r3["normal"]["gold"],
            "r3_normal_answer": answer(r3, "normal"), "r3_text_answer": answer(r3, "text_only"), "r3_wrong_answer": answer(r3, "wrong_image"), "r3_blank_answer": answer(r3, "blank_image"),
            "step25_normal_answer": answer(s25, "normal"), "step25_text_answer": answer(s25, "text_only"), "step25_wrong_answer": answer(s25, "wrong_image"), "step25_blank_answer": answer(s25, "blank_image"),
            "step50_normal_answer": answer(s50, "normal"), "step50_text_answer": answer(s50, "text_only"), "step50_wrong_answer": answer(s50, "wrong_image"), "step50_blank_answer": answer(s50, "blank_image"),
            "r3_case_gap": r3_gap, "step25_case_gap": gap25, "step50_case_gap": gap50,
            "delta_gap_25": delta25, "delta_gap_50": delta50,
            "main_regression_setting": setting, "regression_reason": reason, "suggested_pair_type": pair_type,
        })
    rows.sort(key=lambda row: min(float(row["delta_gap_25"]), float(row["delta_gap_50"])))
    top = rows[: args.limit]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(top)
    setting_counts = Counter(str(row["main_regression_setting"]) for row in top)
    reason_counts = Counter(str(row["regression_reason"]) for row in top)
    lines = [
        "# Stage 8.5 Case Gap Regression",
        "",
        f"- Compared aligned held-out cases: {len(ids)}",
        f"- Cases with regression in step-25 or step-50: {len(rows)}",
        f"- Reported worst cases: {len(top)}",
        "",
        "## Main Regression Setting (Top Cases)",
        "",
    ]
    lines += [f"- `{key}`: {value}" for key, value in setting_counts.most_common()]
    lines += ["", "## Regression Reasons", ""]
    lines += [f"- `{key}`: {value}" for key, value in reason_counts.most_common()]
    lines += ["", "The report is answer-only and uses held-out predictions only; no new inference is performed."]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aligned_cases": len(ids), "regressed_cases": len(rows), "reported": len(top), "main_settings": dict(setting_counts), "reasons": dict(reason_counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
