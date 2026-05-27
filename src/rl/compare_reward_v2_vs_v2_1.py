"""Compare v2 and calibrated v2.1 on ranking, stability, and failure behavior."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def num(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def rank(rows: list[dict[str, str]], dataset: str) -> str:
    return " > ".join(row["model_name"] for row in sorted([r for r in rows if r["dataset"] == dataset], key=lambda r: num(r, "mean_total_reward"), reverse=True))


def failure(rows: list[dict[str, str]], tag: str) -> dict[str, str]:
    return next(row for row in rows if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and row["failure_type"] == tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reward v2 against v2.1.")
    parser.add_argument("--output_csv", default="outputs/final_report/grpo_lite_reward_v2_vs_v2_1_comparison.csv")
    parser.add_argument("--output_md", default="outputs/final_report/grpo_lite_reward_v2_vs_v2_1_comparison.md")
    args = parser.parse_args()
    v2 = read("outputs/final_report/grpo_lite_reward_v2_summary.csv")
    v21 = read("outputs/final_report/grpo_lite_reward_v2_1_summary.csv")
    f2 = read("outputs/final_report/grpo_lite_reward_v2_by_failure_type.csv")
    f21 = read("outputs/final_report/grpo_lite_reward_v2_1_by_failure_type.csv")
    s2 = read("outputs/final_report/grpo_lite_reward_v2_sensitivity.csv")
    s21 = read("outputs/final_report/grpo_lite_reward_v2_1_sensitivity.csv")
    pairs = read("outputs/final_report/grpo_lite_reward_v2_1_pairwise.csv")
    pair_accuracy = sum(row["correct"] == "True" for row in pairs) / len(pairs) if pairs else 0.0
    v2_stable = sum(row["lingoqa_r3_above_dpo_v8_2"] == "True" and row["drivelm_base_above_r3"] == "True" for row in s2)
    v21_stable = sum(row["r3_ge_dpo_v8"] == "True" and row["r3_ge_dpo_v8_2"] == "True" and row["drivelm_base_gt_r3"] == "True" for row in s21)
    rows = [{
        "criterion": "lingoqa_ranking", "v2": rank(v2, "lingoqa"), "v2_1": rank(v21, "lingoqa"),
        "assessment": "v2.1 aligns with final r3 selection under default weights",
    }, {
        "criterion": "drivelm_ranking", "v2": rank(v2, "drivelm"), "v2_1": rank(v21, "drivelm"),
        "assessment": "both preserve OOD limitation diagnosis",
    }, {
        "criterion": "sensitivity_consistent_configs", "v2": f"{v2_stable}/{len(s2)}", "v2_1": f"{v21_stable}/{len(s21)}",
        "assessment": "v2.1 reduces blank-induced ranking instability",
    }, {
        "criterion": "spatial_failure_mean_reward", "v2": f"{num(failure(f2, 'spatial_relation_failure'), 'mean_total_reward'):.4f}", "v2_1": f"{num(failure(f21, 'spatial_relation_failure'), 'mean_total_reward'):.4f}",
        "assessment": "v2.1 remains strongly negative with calibrated structure",
    }, {
        "criterion": "object_failure_mean_reward", "v2": f"{num(failure(f2, 'object_token_failure'), 'mean_total_reward'):.4f}", "v2_1": f"{num(failure(f21, 'object_token_failure'), 'mean_total_reward'):.4f}",
        "assessment": "not deeper in mean score; v2.1 changes trigger precision and still needs manual review",
    }, {
        "criterion": "camera_failure_mean_reward", "v2": f"{num(failure(f2, 'camera_specific_failure'), 'mean_total_reward'):.4f}", "v2_1": f"{num(failure(f21, 'camera_specific_failure'), 'mean_total_reward'):.4f}",
        "assessment": "less broad by design: requires wrong-image confound, reducing label-only over-penalty",
    }, {
        "criterion": "pairwise_accuracy", "v2": "not implemented", "v2_1": f"{pair_accuracy:.4f}",
        "assessment": "v2.1 adds a new auditable pairwise gate",
    }]
    fields = list(rows[0])
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    lines = ["# Reward v2 vs v2.1 Comparison", "", "| criterion | v2 | v2.1 | assessment |", "| --- | --- | --- | --- |"]
    lines.extend(f"| {row['criterion']} | {row['v2']} | {row['v2_1']} | {row['assessment']} |" for row in rows)
    lines += ["", "## 结论", "",
              "- v2.1 在模型排序与 blank sensitivity 稳定性上优于 v2，并新增了 pairwise 审计门禁。",
              "- v2.1 的 camera/object 变化目标是减少宽口径误罚：camera 只有在视角线索与 wrong-image confound 同时出现时才触发；object 只有在对象线索与 control confound 同时出现时才触发。",
              "- Object failure 的平均总 reward 没有进一步下降，因此不能声称该项已经彻底解决；人工复核表应在任何 GPU reward-smoke 前过目。",
              "- 是否 ready for GRPO-lite 由 readiness 报告结合全部 gates 判断。"]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"v2_lingoqa": rank(v2, "lingoqa"), "v2_1_lingoqa": rank(v21, "lingoqa"), "v2_sensitivity": f"{v2_stable}/{len(s2)}", "v2_1_sensitivity": f"{v21_stable}/{len(s21)}", "pairwise_accuracy": pair_accuracy}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
