"""Build a compact human-review sheet for reward disagreements."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PRIORITY = {
    "r3_lower_than_dpo_v8_1": 0,
    "object_failure_under_penalized": 1,
    "blank_penalty_over_sensitive": 2,
    "reward_high_but_case_gap_bad": 3,
    "camera_failure_under_penalized": 4,
    "dpo_v8_2_over_r3_under_stronger_blank": 5,
    "reward_low_but_metrics_ok": 6,
    "normal_refusal_sanity": 7,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reward v2 manual review sheet.")
    parser.add_argument("--input", default="outputs/final_report/reward_v2_disagreement_cases.csv")
    parser.add_argument("--output_csv", default="outputs/final_report/reward_v2_manual_review_sheet.csv")
    parser.add_argument("--output_md", default="outputs/final_report/reward_v2_manual_review_sheet.md")
    parser.add_argument("--limit_per_type", type=int, default=12)
    args = parser.parse_args()
    with Path(args.input).open("r", encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
    source.sort(key=lambda row: (PRIORITY.get(row["disagreement_type"], 99), float(row["total_reward"])))
    counts = Counter()
    rows = []
    for item in source:
        dtype = item["disagreement_type"]
        if counts[dtype] >= args.limit_per_type:
            continue
        counts[dtype] += 1
        rows.append({
            "id": item["id"], "dataset": item["dataset"], "model_name": item["model_name"],
            "question": item["question"], "gold": item["gold"], "normal_answer": item["normal_answer"],
            "text_only_answer": item["text_only_answer"], "wrong_image_answer": item["wrong_image_answer"],
            "blank_image_answer": item["blank_image_answer"], "case_gap": item["case_gap"],
            "total_reward": item["total_reward"], "disagreement_type": dtype,
            "auto_suspected_reason": item["suspected_reason"], "human_label": "", "human_note": "",
            "recommended_reward_fix": item["suggested_fix"],
        })
    fields = list(rows[0]) if rows else ["id"]
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Reward v2 Manual Review Sheet", "",
        "请在 CSV 的 `human_label` 中填写：`reward_correct`、`reward_over_penalized`、`reward_under_penalized`、`metric_conflict` 或 `ambiguous`。",
        "", f"- 待复核行数：{len(rows)}。", "- 优先级：r3/v8.1 排序冲突 > object penalty > blank sensitivity > 高 reward 低 case-gap。",
        "", "| disagreement type | selected rows |", "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in counts.items())
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"manual_review_rows": len(rows), "counts": dict(counts)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
