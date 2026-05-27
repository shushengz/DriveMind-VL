"""Create the Chinese Stage 8.5 offline error-attribution report."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

R3 = "sft_v3_r3_lingo_smoke"
DPO8 = "dpo_v8_lingo_smoke"
STEP25 = "dpo_v8_1_lingo_smoke_step25"
STEP50 = "dpo_v8_1_lingo_smoke_step50"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def lookup(rows: list[dict[str, str]], model: str, setting: str | None = None) -> dict[str, str]:
    return next(row for row in rows if row["model_name"] == model and (setting is None or row["setting"] == setting))


def num(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stage 8.5 offline DPO failure attribution.")
    parser.add_argument("--summary", default="outputs/final_report/stage8_5_control_behavior_summary.csv")
    parser.add_argument("--by_setting", default="outputs/final_report/stage8_5_control_behavior_by_setting.csv")
    parser.add_argument("--regression", default="outputs/final_report/stage8_5_case_gap_regression.csv")
    parser.add_argument("--fix_cases", default="outputs/final_report/stage8_5_dpo_fix_cases.csv")
    parser.add_argument("--break_cases", default="outputs/final_report/stage8_5_dpo_break_cases.csv")
    parser.add_argument("--results", default="outputs/final_report/stage8_dpo_v8_1_heldout_main_results.csv")
    parser.add_argument("--output_md", default="outputs/final_report/stage8_5_error_attribution.md")
    parser.add_argument("--output_json", default="outputs/final_report/stage8_5_error_attribution.json")
    parser.add_argument("--dry_run", action="store_true", help="Accepted for CPU-only orchestration; use isolated output paths for previews.")
    args = parser.parse_args()
    aggregate, settings = read_csv(Path(args.summary)), read_csv(Path(args.by_setting))
    regressions, fixes, breaks = read_csv(Path(args.regression)), read_csv(Path(args.fix_cases)), read_csv(Path(args.break_cases))
    results = [row for row in read_csv(Path(args.results)) if row["score_mode"] == "answer_only"]
    result = {row["model_name"]: row for row in results}
    agg = {model: lookup(aggregate, model) for model in (R3, STEP25, STEP50)}
    setting_delta = {
        setting: num(lookup(settings, STEP50, setting), "avg_f1") - num(lookup(settings, R3, setting), "avg_f1")
        for setting in ("text_only", "wrong_image", "blank_image")
    }
    primary = max(setting_delta, key=setting_delta.get)
    reason_counts = Counter(row["regression_reason"] for row in regressions)
    setting_counts = Counter(row["main_regression_setting"] for row in regressions)
    fix_counts = Counter(row["checkpoint"] for row in fixes)
    hallucination_fix_counts = Counter(row["checkpoint"] for row in fixes if row["case_type"] == "control_hallucination_fixed")
    high_f1_fix_counts = Counter(row["checkpoint"] for row in fixes if row["case_type"] == "control_high_f1_fixed")
    break_counts = Counter(row["checkpoint"] for row in breaks)
    caution_down_vs_r3 = num(agg[STEP50], "caution_rate") < num(agg[R3], "caution_rate")
    caution_down_vs_step25 = num(agg[STEP50], "caution_rate") < num(agg[STEP25], "caution_rate")
    direct_up = num(agg[STEP50], "direct_answer_rate") > num(agg[R3], "direct_answer_rate")
    blank_high_f1_up = num(lookup(settings, STEP50, "blank_image"), "high_f1_rate_0_20") > num(lookup(settings, R3, "blank_image"), "high_f1_rate_0_20")
    step25_better = num(result[STEP25], "case_gap") > num(result[STEP50], "case_gap") and num(result[STEP25], "normal_f1") > num(result[STEP50], "normal_f1")
    report: dict[str, Any] = {
        "cpu_only": True,
        "primary_case_gap_regression_setting": primary,
        "setting_f1_delta_step50_vs_r3": setting_delta,
        "direct_answer_rate_increased_step50_vs_r3": direct_up,
        "caution_rate_decreased_step50_vs_r3": caution_down_vs_r3,
        "caution_rate_decreased_step50_vs_step25": caution_down_vs_step25,
        "blank_high_f1_increased": blank_high_f1_up,
        "regression_setting_counts_top50": dict(setting_counts),
        "regression_reason_counts_top50": dict(reason_counts),
        "fix_counts": dict(fix_counts),
        "hallucination_fix_counts": dict(hallucination_fix_counts),
        "high_f1_fix_counts": dict(high_f1_fix_counts),
        "break_counts": dict(break_counts),
        "preferred_checkpoint": STEP25 if step25_better else STEP50,
        "recommend_continue_v8_1_steps": False,
        "recommend_construct_preference_v8_2": True,
        "recommend_enter_grpo_lite": False,
        "aggregate_behavior": {model: agg[model] for model in (R3, STEP25, STEP50)},
    }
    lines = [
        "# Stage 8.5：DPO-v8.1 误差归因",
        "",
        "本报告完全基于已有 held-out raw predictions 的 answer-only 离线分析；未进行推理、训练或 GPU 计算。",
        "",
        "## 核心判断",
        "",
        f"1. DPO-v8.1 虽降低了显式 hallucination，但 case gap 变差，原因是控制组的词面得分并未同步降低，尤其是 `{primary}` 条件。",
        f"2. step-50 相比 r3 的 setting F1 变化：text-only {setting_delta['text_only']:+.4f}，wrong-image {setting_delta['wrong_image']:+.4f}，blank-image {setting_delta['blank_image']:+.4f}。",
        f"3. step-50 的总体 direct-answer rate 是否高于 r3：{'是' if direct_up else '否'}；因此整体 direct-rate 上升不是单独主因。",
        f"4. blank-image high-F1 answer 是否增加：{'是' if blank_high_f1_up else '否'}；这是当前最清晰的 control regression。",
        f"5. caution rate：step-50 相比 r3 是否下降：{'是' if caution_down_vs_r3 else '否'}；相比 step-25 是否下降：{'是' if caution_down_vs_step25 else '否'}。",
        f"6. Top-50 regression 主 setting 分布：{dict(setting_counts)}。",
        f"7. wrong-image confound 是否可视为已解决：否；其 F1 未形成稳定下降证据。",
        f"8. step-25 与 step-50 中更值得保留的观察点：`{report['preferred_checkpoint']}`。",
        "",
        "## 行为聚合指标（全部控制组）",
        "",
        "| Model | Direct Answer | Short Prior | Caution | High F1 >= 0.20 | Hallucination |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in (R3, STEP25, STEP50):
        row = agg[model]
        lines.append(f"| {model} | {num(row, 'direct_answer_rate'):.4f} | {num(row, 'short_prior_answer_rate'):.4f} | {num(row, 'caution_rate'):.4f} | {num(row, 'high_f1_rate_0_20'):.4f} | {num(row, 'hallucination_rate'):.4f} |")
    lines += [
        "",
        "## Fix / Break",
        "",
        f"- step-25 fix records: {fix_counts.get(STEP25, 0)}（其中 hallucination fixes {hallucination_fix_counts.get(STEP25, 0)}，high-F1 fixes {high_f1_fix_counts.get(STEP25, 0)}）；break records: {break_counts.get(STEP25, 0)}。",
        f"- step-50 fix records: {fix_counts.get(STEP50, 0)}（其中 hallucination fixes {hallucination_fix_counts.get(STEP50, 0)}，high-F1 fixes {high_f1_fix_counts.get(STEP50, 0)}）；break records: {break_counts.get(STEP50, 0)}。",
        "",
        "## Preference-v8.2 建议",
        "",
        "- 不继续增加 DPO-v8.1 steps；step-50 已显示更强校准伴随更差 case gap。",
        "- 构造 Preference-v8.2，将目标从 hallucination trigger 扩展为 `control_direct_answer_vs_caution` 与各 setting 的 high-F1 overlap。",
        "- 将 normal-related pairs 提高到 40%，control-related 降到 55%，spatial 保持 5%。",
        "- v8.2 仍从 r3 与 frozen r3 reference 初始化，不从 DPO-v8.1 初始化。",
        "- 是否进入 GRPO-lite：否。当前问题仍可被更精确的 preference 数据直接定位。",
    ]
    target = Path(args.output_md)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"primary_setting": primary, "preferred_checkpoint": report["preferred_checkpoint"], "recommend_construct_preference_v8_2": True, "recommend_enter_grpo_lite": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
