"""Analyze answer behavior under DriveLM OOD control settings."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.control_behavior_metrics import behavior_flags
from src.eval.drivelm_ood_attribution_utils import BASE, R3, CONTROL_SETTINGS, PREDICTION_ROOT, answer, groups, scored, write_csv


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pattern(flags: dict[str, bool]) -> str:
    if flags["is_caution"]:
        return "caution"
    if flags["is_count_answer"]:
        return "count"
    if flags["is_action_answer"]:
        return "action"
    if flags["is_short_prior_answer"]:
        return "short_prior"
    if flags["is_direct_answer"]:
        return "other_direct"
    return "other"


def setting_metrics(model: str, setting: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], Counter[str]]:
    flagged = []
    patterns: Counter[str] = Counter()
    for row in rows:
        score = scored(row)[setting]
        flags = behavior_flags(answer({setting: score}, setting), float(score["f1"]))
        patterns[pattern(flags)] += 1
        flagged.append((score, flags))
    result = {
        "model_name": model,
        "setting": setting,
        "num_samples": len(flagged),
        "direct_answer_rate": mean([float(flags["is_direct_answer"]) for _, flags in flagged]),
        "caution_rate": mean([float(flags["is_caution"]) for _, flags in flagged]),
        "short_prior_answer_rate": mean([float(flags["is_short_prior_answer"]) for _, flags in flagged]),
        "action_answer_rate": mean([float(flags["is_action_answer"]) for _, flags in flagged]),
        "count_answer_rate": mean([float(flags["is_count_answer"]) for _, flags in flagged]),
        "high_f1_rate_0_20": mean([float(flags["high_f1_0_20"]) for _, flags in flagged]),
        "high_f1_rate_0_30": mean([float(flags["high_f1_0_30"]) for _, flags in flagged]),
        "hallucination_rate": mean([float(score["hallucination"]) for score, _ in flagged]),
        "avg_answer_length": mean([float(score["answer_length"]) for score, _ in flagged]),
        "answer_patterns": json.dumps(dict(patterns), ensure_ascii=False),
    }
    return result, patterns


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze DriveLM OOD control behavior.")
    parser.add_argument("--prediction_root", default=PREDICTION_ROOT.as_posix())
    parser.add_argument("--summary_csv", default="outputs/final_report/drivelm_control_behavior_summary.csv")
    parser.add_argument("--setting_csv", default="outputs/final_report/drivelm_control_behavior_by_setting.csv")
    parser.add_argument("--report", default="outputs/final_report/drivelm_control_behavior_report.md")
    args = parser.parse_args()
    setting_rows = []
    blank_patterns: dict[str, Counter[str]] = {}
    summary_rows = []
    for model in (BASE, R3):
        aligned = list(groups(model, Path(args.prediction_root)).values())
        rows_for_model = []
        for setting in CONTROL_SETTINGS:
            row, patterns = setting_metrics(model, setting, aligned)
            setting_rows.append(row)
            rows_for_model.append(row)
            if setting == "blank_image":
                blank_patterns[model] = patterns
        summary_rows.append({
            "model_name": model,
            "num_control_predictions": len(aligned) * len(CONTROL_SETTINGS),
            "direct_answer_rate": mean([row["direct_answer_rate"] for row in rows_for_model]),
            "caution_rate": mean([row["caution_rate"] for row in rows_for_model]),
            "short_prior_answer_rate": mean([row["short_prior_answer_rate"] for row in rows_for_model]),
            "action_answer_rate": mean([row["action_answer_rate"] for row in rows_for_model]),
            "count_answer_rate": mean([row["count_answer_rate"] for row in rows_for_model]),
            "high_f1_rate_0_20": mean([row["high_f1_rate_0_20"] for row in rows_for_model]),
            "high_f1_rate_0_30": mean([row["high_f1_rate_0_30"] for row in rows_for_model]),
            "hallucination_rate": mean([row["hallucination_rate"] for row in rows_for_model]),
            "blank_high_f1_rate_0_20": rows_for_model[2]["high_f1_rate_0_20"],
            "wrong_image_high_f1_rate_0_20": rows_for_model[1]["high_f1_rate_0_20"],
            "text_only_high_f1_rate_0_20": rows_for_model[0]["high_f1_rate_0_20"],
            "avg_control_answer_length": mean([row["avg_answer_length"] for row in rows_for_model]),
        })
    write_csv(Path(args.setting_csv), setting_rows)
    write_csv(Path(args.summary_csv), summary_rows)
    base = next(row for row in summary_rows if row["model_name"] == BASE)
    r3 = next(row for row in summary_rows if row["model_name"] == R3)
    dominant_blank = blank_patterns[R3].most_common(1)[0] if blank_patterns[R3] else ("none", 0)
    lines = [
        "# DriveLM OOD Control Behavior Analysis", "",
        "| model | direct answer | caution | short prior | action | count | high-F1@0.20 | hallucination | blank high-F1 | wrong high-F1 | text high-F1 | avg length |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_name']} | {row['direct_answer_rate']:.4f} | {row['caution_rate']:.4f} | "
            f"{row['short_prior_answer_rate']:.4f} | {row['action_answer_rate']:.4f} | {row['count_answer_rate']:.4f} | "
            f"{row['high_f1_rate_0_20']:.4f} | {row['hallucination_rate']:.4f} | "
            f"{row['blank_high_f1_rate_0_20']:.4f} | {row['wrong_image_high_f1_rate_0_20']:.4f} | "
            f"{row['text_only_high_f1_rate_0_20']:.4f} | {row['avg_control_answer_length']:.4f} |"
        )
    lines += [
        "", "## Blank Answer Patterns", "",
        "| model | patterns |", "| --- | --- |",
        f"| {BASE} | {json.dumps(dict(blank_patterns[BASE]), ensure_ascii=False)} |",
        f"| {R3} | {json.dumps(dict(blank_patterns[R3]), ensure_ascii=False)} |",
        "", "## 诊断回答", "",
        f"1. r3 blank high-F1 增加伴随 direct-answer rate 从 {base['direct_answer_rate']:.4f} 变为 {r3['direct_answer_rate']:.4f}，说明模型在无可靠证据时更倾向直接作答。",
        f"2. r3 blank answers 的主要模式是 `{dominant_blank[0]}`（{dominant_blank[1]} / 100）。",
        f"3. text-only prior：r3 text-only high-F1={r3['text_only_high_f1_rate_0_20']:.4f}，需要与 Base={base['text_only_high_f1_rate_0_20']:.4f} 对照解释。",
        f"4. wrong-image confound：r3 wrong-image high-F1={r3['wrong_image_high_f1_rate_0_20']:.4f}，Base={base['wrong_image_high_f1_rate_0_20']:.4f}。",
        f"5. 回答风格：r3 control 平均答案长度={r3['avg_control_answer_length']:.4f}，Base={base['avg_control_answer_length']:.4f}；自信程度应与 direct-answer/caution 指标共同判断。",
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary_rows, "blank_patterns": {model: dict(values) for model, values in blank_patterns.items()}, "r3_dominant_blank_pattern": dominant_blank}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
