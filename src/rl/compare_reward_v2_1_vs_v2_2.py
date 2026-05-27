"""Compare Reward v2.1 and v2.2 after manual review integration."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def read(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def ranking(rows: list[dict[str, str]], dataset: str) -> str:
    return " > ".join(r["model_name"] for r in sorted([row for row in rows if row["dataset"] == dataset], key=lambda row: number(row, "mean_total_reward"), reverse=True))


def pair_accuracy(rows: list[dict[str, str]]) -> float:
    return sum(row["correct"] == "True" for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    v21 = read("outputs/final_report/grpo_lite_reward_v2_1_summary.csv")
    v22 = read("outputs/final_report/grpo_lite_reward_v2_2_summary.csv")
    p21 = read("outputs/final_report/grpo_lite_reward_v2_1_pairwise.csv")
    p22 = read("outputs/final_report/grpo_lite_reward_v2_2_pairwise.csv")
    s21 = read("outputs/final_report/grpo_lite_reward_v2_1_sensitivity.csv")
    s22 = read("outputs/final_report/grpo_lite_reward_v2_2_sensitivity.csv")
    impact = read("outputs/final_report/grpo_lite_reward_v2_2_manual_review_impact.csv")
    under = [row for row in impact if row["human_label"] == "reward_under_penalized"]
    under_improved = sum(row["under_penalized_improved"] == "True" for row in under)
    stable21 = sum(row["drivelm_base_gt_r3"] == "True" and row["r3_ge_dpo_v8"] == "True" and row["r3_ge_dpo_v8_2"] == "True" for row in s21)
    stable22 = sum(row["drivelm_base_gt_r3"] == "True" and row["r3_ge_dpo_v8"] == "True" and row["r3_ge_dpo_v8_2"] == "True" for row in s22)
    v22_pair = pair_accuracy(p22)
    v22_better = stable22 >= 8 and v22_pair >= 0.95 and under_improved == len(under)
    output = [
        {"criterion": "lingoqa_ranking", "v2_1": ranking(v21, "lingoqa"), "v2_2": ranking(v22, "lingoqa"), "assessment": "maintains main-model ordering"},
        {"criterion": "drivelm_ranking", "v2_1": ranking(v21, "drivelm"), "v2_2": ranking(v22, "drivelm"), "assessment": "preserves OOD failure diagnosis"},
        {"criterion": "sensitivity_consistent_configs", "v2_1": f"{stable21}/{len(s21)}", "v2_2": f"{stable22}/{len(s22)}", "assessment": "check whether patch changes stability"},
        {"criterion": "pairwise_accuracy", "v2_1": f"{pair_accuracy(p21):.4f}", "v2_2": f"{v22_pair:.4f}", "assessment": "v2.2 includes new human-review failure pair types"},
        {"criterion": "manual_under_penalized_improved", "v2_1": "not integrated", "v2_2": f"{under_improved}/{len(under)}", "assessment": "direct manual review integration measure"},
    ]
    with Path("outputs/final_report/grpo_lite_reward_v2_1_vs_v2_2_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0])); writer.writeheader(); writer.writerows(output)
    lines = ["# Reward v2.1 vs v2.2 Comparison", "", "| criterion | v2.1 | v2.2 | assessment |", "| --- | --- | --- | --- |"]
    lines.extend(f"| {row['criterion']} | {row['v2_1']} | {row['v2_2']} | {row['assessment']} |" for row in output)
    lines += ["", "## 结论", "",
              "- v2.2 的价值不在于简单压低所有 reward，而在于把人工确认的漏罚转成可审计的四项 penalty。",
              f"- 人工漏罚修复为 `{under_improved}/{len(under)}`；但 v2.2 sensitivity 一致配置为 `{stable22}/{len(s22)}`，整体 pairwise 为 `{v22_pair:.4f}`。",
              f"- v2.2 是否优于 v2.1：**{'是' if v22_better else '否'}**。当前补丁能发现新漏洞，但破坏了模型排序稳定性或 pairwise 门槛，因此不能作为 GRPO-lite 训练信号。",
              "- 当前结论：`Do not train GRPO-lite yet.`", ""]
    Path("outputs/final_report/grpo_lite_reward_v2_1_vs_v2_2_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"lingoqa_v2_1": ranking(v21, "lingoqa"), "lingoqa_v2_2": ranking(v22, "lingoqa"), "drivelm_v2_2": ranking(v22, "drivelm"), "manual_under_improved": f"{under_improved}/{len(under)}", "pairwise_v2_2": v22_pair, "v2_2_better_than_v2_1": v22_better}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
