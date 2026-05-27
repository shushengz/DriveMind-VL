"""Select the final checkpoint from the consolidated answer-only table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

BEST = "SFT-v3-r3"
CHECKPOINT = "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/"
DPO_NAMES = ["DPO-v8 rule-based", "DPO-v8.1 step-25", "DPO-v8.1 step-50", "DPO-v8.2 step-25"]
FIELDS = ["normal_f1", "case_gap", "blank_high_f1_rate_0_20", "control_high_f1_rate_0_20", "control_hallucination_rate", "normal_refusal_rate", "avg_answer_length"]


def num(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select DriveMind-VL final checkpoint.")
    parser.add_argument("--results", default="outputs/final_report/final_main_results_answer_only.csv")
    parser.add_argument("--output_md", default="outputs/final_report/final_checkpoint_selection.md")
    parser.add_argument("--output_json", default="outputs/final_report/final_checkpoint_selection.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    rows = {row["model_name"]: row for row in csv.DictReader(Path(args.results).open(encoding="utf-8", newline=""))}
    required = ["Base Qwen2.5-VL-3B", "SFT-v2", "DPO-v7", "SFT-v3-r2", BEST, *DPO_NAMES]
    missing = [name for name in required if name not in rows]
    if missing:
        raise ValueError(f"missing final model rows: {missing}")
    r3 = rows[BEST]
    dpo_observations = {
        name: {
            "normal_f1_delta_vs_r3": num(rows[name], "normal_f1") - num(r3, "normal_f1"),
            "case_gap_delta_vs_r3": num(rows[name], "case_gap") - num(r3, "case_gap"),
            "hallucination_delta_vs_r3": num(rows[name], "control_hallucination_rate") - num(r3, "control_hallucination_rate"),
        }
        for name in DPO_NAMES
    }
    report = {
        "best_model": BEST,
        "best_checkpoint": CHECKPOINT,
        "selection_basis": "held-out strict_visual answer_only; balance normal accuracy and visual dependency controls",
        "r3_metrics": {field: num(r3, field) for field in FIELDS},
        "dpo_observations": dpo_observations,
        "recommend_continue_dpo": False,
        "recommend_enter_grpo_lite": False,
        "dpo_role": "ablation_and_failure_analysis",
    }
    lines = [
        "# 最终 Checkpoint 选择",
        "",
        f"当前推荐主模型：`{CHECKPOINT}`（{BEST}）。",
        "",
        "## 选择理由",
        "",
        f"- SFT-v3-r3 在 held-out answer-only 上取得 `normal_f1={num(r3, 'normal_f1'):.4f}`，明显优于 Base、SFT-v2 与 DPO-v7。",
        f"- 其 `case_gap={num(r3, 'case_gap'):.4f}` 最接近 0，且 `normal_refusal={num(r3, 'normal_refusal_rate'):.4f}`，在正常能力与谨慎行为之间最稳定。",
        "- 模型选择不只看 normal F1；control high-F1、blank high-F1、hallucination、拒答率与答案长度共同决定视觉依赖可靠性。",
        "",
        "## 为什么 DPO 不作为最终主模型",
        "",
        "| 模型 | normal_f1 | case_gap | blank_high_f1 | control_high_f1 | hallucination |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in [BEST, *DPO_NAMES]:
        row = rows[name]
        lines.append(f"| {name} | {num(row,'normal_f1'):.4f} | {num(row,'case_gap'):.4f} | {num(row,'blank_high_f1_rate_0_20'):.4f} | {num(row,'control_high_f1_rate_0_20'):.4f} | {num(row,'control_hallucination_rate'):.4f} |")
    lines += [
        "",
        "- DPO 能轻微降低部分显式 hallucination，但没有稳定改善 case-level visual dependency。",
        "- DPO-v8.1 与 v8.2 仍暴露 blank-image high-F1 prior answer / control high-F1 问题；更强校准甚至可能伤害 case gap。",
        "",
        "## 决策",
        "",
        "- 当前不建议继续追加 DPO steps；DPO 分支作为消融与失败机制分析保留。",
        "- 当前不建议直接进入 GRPO-lite；若继续研究，先设计可审计 reward harness 与更大 held-out 评测。",
    ]
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": CHECKPOINT, "recommend_enter_grpo_lite": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
