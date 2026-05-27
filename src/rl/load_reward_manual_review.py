"""Load the reviewed reward disagreement sheet and summarize human labels."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/final_report/reward_v2_manual_review_sheet_reviewed.csv")
    parser.add_argument("--json", default="outputs/final_report/reward_manual_review_summary.json")
    parser.add_argument("--md", default="outputs/final_report/reward_manual_review_summary.md")
    args = parser.parse_args()
    path = Path(args.input)
    if not path.exists():
        summary = {"manual_review_file_found": False, "warning": "manual review file missing; GRPO-lite readiness must fail", "total_review_rows": 0}
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        Path(args.md).write_text("# Reward Manual Review Summary\n\n警告：人工复核表缺失，不能进入 GRPO-lite。\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = Counter((row.get("human_label") or "").strip() for row in rows)
    disagreements = Counter((row.get("disagreement_type") or "").strip() for row in rows)
    fixes = Counter((row.get("recommended_reward_fix") or "").strip() for row in rows)
    needed = []
    joined_fixes = "\n".join(fixes)
    for marker, name in (
        ("object_token_invariant", "object_token_invariant_answer_penalty"),
        ("normal_object_category_mismatch", "normal_object_category_mismatch_penalty"),
        ("invalid_generic", "invalid_generic_answer_penalty"),
        ("control_same_as_normal", "control_same_as_normal_penalty"),
    ):
        if marker in joined_fixes or any(marker.replace("_", " ") in (row.get("human_note") or "") for row in rows):
            needed.append(name)
    summary = {
        "manual_review_file_found": True,
        "source_file": str(path),
        "total_review_rows": len(rows),
        "human_label_counts": dict(labels),
        "disagreement_type_counts": dict(disagreements),
        "reward_under_penalized_cases": labels.get("reward_under_penalized", 0),
        "reward_over_penalized_cases": labels.get("reward_over_penalized", 0),
        "metric_conflict_cases": labels.get("metric_conflict", 0),
        "reward_correct_cases": labels.get("reward_correct", 0),
        "recommended_reward_fix_counts": dict(fixes),
        "v2_2_required_patches": needed,
        "blocking_missing_label_count": labels.get("", 0),
    }
    Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Reward Manual Review Summary", "",
        f"- 已读取人工复核表：`{path}`。",
        f"- 总复核条目：{len(rows)}。",
        "", "## Human Labels", "", "| label | count |", "| --- | ---: |",
    ]
    lines.extend(f"| {label or '(blank)'} | {count} |" for label, count in labels.most_common())
    lines += ["", "## v2.2 需要处理的问题", ""]
    lines.extend(f"- `{patch}`" for patch in needed)
    lines += ["", "人工意见表明 v2.1 不能直接用于训练；v2.2 必须先在同一离线记录上重新通过审计。", ""]
    Path(args.md).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
