"""Measure whether Reward v2.2 addresses the human-reviewed v2.1 gaps."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def read(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", default="outputs/final_report/reward_v2_manual_review_sheet_reviewed.csv")
    parser.add_argument("--v21", default="outputs/final_report/grpo_lite_reward_v2_1_debug.csv")
    parser.add_argument("--v22", default="outputs/final_report/grpo_lite_reward_v2_2_debug.csv")
    parser.add_argument("--csv", default="outputs/final_report/grpo_lite_reward_v2_2_manual_review_impact.csv")
    parser.add_argument("--md", default="outputs/final_report/grpo_lite_reward_v2_2_manual_review_impact.md")
    args = parser.parse_args()
    reviewed = read(args.review)
    old = {(r["id"], r["dataset"], r["model_name"]): r for r in read(args.v21)}
    new = {(r["id"], r["dataset"], r["model_name"]): r for r in read(args.v22)}
    rows = []
    missing = []
    for item in reviewed:
        key = (item["id"], item["dataset"], item["model_name"])
        if key not in old or key not in new:
            missing.append("|".join(key))
            continue
        v21, v22 = old[key], new[key]
        delta = value(v22, "total_reward") - value(v21, "total_reward")
        patch_components = {
            "object_invariant": value(v22, "object_token_invariant_penalty"),
            "category_mismatch": value(v22, "normal_object_category_mismatch_penalty"),
            "invalid_generic": value(v22, "invalid_generic_answer_penalty"),
            "same_as_normal": value(v22, "control_same_as_normal_penalty"),
        }
        patched = [name for name, amount in patch_components.items() if amount > 0]
        label = item.get("human_label", "")
        improved = label == "reward_under_penalized" and delta <= -0.10
        over_harmed = label == "reward_over_penalized" and delta < -0.10
        rows.append({
            "id": item["id"], "dataset": item["dataset"], "model_name": item["model_name"],
            "human_label": label, "disagreement_type": item.get("disagreement_type", ""),
            "recommended_reward_fix": item.get("recommended_reward_fix", ""),
            "v2_1_reward": value(v21, "total_reward"), "v2_2_reward": value(v22, "total_reward"), "reward_delta": delta,
            **{f"{name}_penalty": amount for name, amount in patch_components.items()},
            "patches_triggered": "|".join(patched), "under_penalized_improved": improved, "over_penalized_further_harmed": over_harmed,
        })
    with Path(args.csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["id"]); writer.writeheader(); writer.writerows(rows)
    buckets: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["human_label"])].append(row)
    under = buckets.get("reward_under_penalized", [])
    over = buckets.get("reward_over_penalized", [])
    fixed = sum(bool(row["under_penalized_improved"]) for row in under)
    patch_count = Counter(patch for row in rows for patch in str(row["patches_triggered"]).split("|") if patch)
    lines = ["# Reward v2.2 Manual Review Impact", "",
             f"- 已匹配人工复核记录：{len(rows)}/{len(reviewed)}；缺失：{len(missing)}。",
             f"- `reward_under_penalized` 改善：{fixed}/{len(under)}（定义为 reward 至少再下降 0.10）。",
             f"- `reward_over_penalized` 继续明显受损：{sum(bool(row['over_penalized_further_harmed']) for row in over)}/{len(over)}。",
             "", "## 按人工标签统计", "", "| label | n | mean reward delta |", "| --- | ---: | ---: |"]
    for label, items in buckets.items():
        lines.append(f"| {label} | {len(items)} | {mean(float(row['reward_delta']) for row in items):.4f} |")
    lines += ["", "## Patch 触发统计", "", "| patch | reviewed rows triggered |", "| --- | ---: |"]
    lines.extend(f"| {patch} | {count} |" for patch, count in patch_count.most_common())
    if missing:
        lines += ["", "## Warning", "", "以下复核行无法映射到 debug 输出：" + ", ".join(missing)]
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print({"review_rows": len(reviewed), "matched": len(rows), "under_total": len(under), "under_improved": fixed, "over_harmed": sum(bool(row["over_penalized_further_harmed"]) for row in over), "patch_triggers": dict(patch_count)})


if __name__ == "__main__":
    main()
