"""Diagnose Stage 10 DPO-v8.2 against r3 and previous DPO checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

R3 = "sft_v3_r3_lingo_smoke"
DPO8 = "dpo_v8_lingo_smoke"
V81_25 = "dpo_v8_1_lingo_smoke_step25"
V81_50 = "dpo_v8_1_lingo_smoke_step50"
V82 = "dpo_v8_2_lingo_smoke_step25"
COMPARE = [R3, DPO8, V81_25, V81_50, V82]


def load_answer_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["model_name"]: row for row in csv.DictReader(handle) if row.get("score_mode") == "answer_only"}


def value(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def dominates(candidate: dict[str, str], other: dict[str, str]) -> bool:
    return (
        value(candidate, "normal_f1") >= value(other, "normal_f1") - 0.02
        and value(candidate, "case_gap") >= value(other, "case_gap")
        and value(candidate, "blank_high_f1_rate_0_20") <= value(other, "blank_high_f1_rate_0_20")
        and value(candidate, "control_high_f1_rate_0_20") <= value(other, "control_high_f1_rate_0_20")
        and value(candidate, "control_hallucination_rate") <= value(other, "control_hallucination_rate")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stage 10 DPO-v8.2 held-out results.")
    parser.add_argument("--results", default="outputs/final_report/stage10_dpo_v8_2_heldout_main_results.csv")
    parser.add_argument("--train_summary", default="outputs/train_logs/dpo_v8_2_lingo_smoke/train_summary.json")
    parser.add_argument("--audit", default="outputs/data_audit/preference_v8_2_audit.json")
    parser.add_argument("--output_json", default="outputs/final_report/stage10_dpo_v8_2_diagnosis.json")
    parser.add_argument("--output_md", default="outputs/final_report/stage10_dpo_v8_2_diagnosis.md")
    args = parser.parse_args()

    rows = load_answer_rows(Path(args.results))
    missing = [name for name in COMPARE if name not in rows]
    if missing:
        raise ValueError(f"missing answer-only result rows: {missing}")
    summary = json.loads(Path(args.train_summary).read_text(encoding="utf-8"))
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    r3, candidate = rows[R3], rows[V82]

    preserved_normal = value(candidate, "normal_f1") >= value(r3, "normal_f1") - 0.02
    case_gap_improved = value(candidate, "case_gap") >= value(r3, "case_gap")
    blank_reduced = value(candidate, "blank_high_f1_rate_0_20") <= value(r3, "blank_high_f1_rate_0_20")
    control_high_reduced = value(candidate, "control_high_f1_rate_0_20") <= value(r3, "control_high_f1_rate_0_20")
    hallucination_reduced = value(candidate, "control_hallucination_rate") <= value(r3, "control_hallucination_rate")
    refusal_ok = value(candidate, "normal_refusal_rate") <= 0.02
    length_bad = value(candidate, "avg_answer_length") > max(value(r3, "avg_answer_length") * 1.25, value(r3, "avg_answer_length") + 8)
    success = all([preserved_normal, case_gap_improved, blank_reduced, control_high_reduced, refusal_ok, not length_bad])
    excellent = (
        value(candidate, "normal_f1") >= value(r3, "normal_f1")
        and value(candidate, "case_gap") > 0
        and value(candidate, "blank_high_f1_rate_0_20") < value(r3, "blank_high_f1_rate_0_20")
        and value(candidate, "control_hallucination_rate") <= value(rows[V81_50], "control_hallucination_rate")
        and value(candidate, "normal_refusal_rate") == 0
    )
    frozen_r3 = (
        summary.get("reference_free") is False
        and "sft_v3_r3_lingo_smoke" in str(summary.get("init_adapter", ""))
        and summary.get("init_adapter") == summary.get("reference_adapter")
    )
    trained = summary.get("success") is True and not summary.get("nan_detected", False) and not summary.get("oom", False)
    failure = (
        value(candidate, "normal_f1") < value(r3, "normal_f1") - 0.02
        or value(candidate, "case_gap") < value(r3, "case_gap")
        or value(candidate, "blank_high_f1_rate_0_20") > value(r3, "blank_high_f1_rate_0_20")
        or value(candidate, "control_high_f1_rate_0_20") > value(r3, "control_high_f1_rate_0_20")
        or value(candidate, "normal_refusal_rate") > value(r3, "normal_refusal_rate")
    )
    better_v8 = dominates(candidate, rows[DPO8])
    better_v81 = dominates(candidate, rows[V81_25]) and dominates(candidate, rows[V81_50])

    if failure or not trained:
        best = R3
        continue_dpo = False
    elif success:
        best = V82
        continue_dpo = False
    else:
        best = max(
            [R3, V82],
            key=lambda name: (
                value(rows[name], "case_gap"),
                -value(rows[name], "blank_high_f1_rate_0_20"),
                -value(rows[name], "control_high_f1_rate_0_20"),
                value(rows[name], "normal_f1"),
            ),
        )
        continue_dpo = False

    report: dict[str, Any] = {
        "training_success": trained,
        "frozen_r3_reference": frozen_r3,
        "uses_case_gap_aware_preference": audit.get("train_ready") is True and audit.get("blank_high_f1_pair_count", 0) > 0,
        "checks": {
            "normal_f1_preserved": preserved_normal,
            "case_gap_improved": case_gap_improved,
            "blank_high_f1_reduced": blank_reduced,
            "control_high_f1_reduced": control_high_reduced,
            "hallucination_reduced": hallucination_reduced,
            "normal_refusal_ok": refusal_ok,
            "answer_length_abnormal": length_bad,
        },
        "stage10_success": success and trained,
        "stage10_excellent": excellent and trained,
        "stage10_failure": failure or not trained,
        "better_than_dpo_v8": better_v8,
        "better_than_dpo_v8_1": better_v81,
        "best_checkpoint": best,
        "recommend_continue_dpo": continue_dpo,
        "recommend_build_v8_3": failure or not success,
        "recommend_enter_grpo_lite": False,
        "train_summary": summary,
        "r3": r3,
        "dpo_v8": rows[DPO8],
        "dpo_v8_1_step25": rows[V81_25],
        "dpo_v8_1_step50": rows[V81_50],
        "dpo_v8_2_step25": candidate,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    verdict = "成功" if report["stage10_success"] else ("失败" if report["stage10_failure"] else "部分成功")
    lines = [
        "# Stage 10：DPO-v8.2 Case-gap-aware GPU Smoke 诊断",
        "",
        f"- 训练是否成功：{'是' if trained else '否'}。",
        f"- 是否使用 frozen r3 reference：{'是' if frozen_r3 else '否'}。",
        f"- 是否使用 case-gap-aware preference：{'是' if report['uses_case_gap_aware_preference'] else '否'}。",
        f"- 总体判定：{verdict}。",
        "",
        "## Answer-only Held-out 核心表",
        "",
        "| 模型 | normal_f1 | case_gap | blank_high_f1 | control_high_f1 | hallucination | normal_refusal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in COMPARE:
        row = rows[name]
        lines.append(
            f"| {name} | {value(row, 'normal_f1'):.4f} | {value(row, 'case_gap'):.4f} | "
            f"{value(row, 'blank_high_f1_rate_0_20'):.4f} | {value(row, 'control_high_f1_rate_0_20'):.4f} | "
            f"{value(row, 'control_hallucination_rate'):.4f} | {value(row, 'normal_refusal_rate'):.4f} |"
        )
    lines += [
        "",
        "## 问题回答",
        "",
        f"1. DPO-v8.2 是否保住 r3 normal F1：{'是' if preserved_normal else '否'}。",
        f"2. 是否改善 r3 case_gap：{'是' if case_gap_improved else '否'}。",
        f"3. 是否降低 blank high-F1 prior answer：{'是' if blank_reduced else '否'}。",
        f"4. 是否降低 control high-F1 rate：{'是' if control_high_reduced else '否'}。",
        f"5. 是否降低 hallucination：{'是' if hallucination_reduced else '否'}。",
        f"6. 是否出现不可接受的 normal refusal：{'否' if refusal_ok else '是'}。",
        f"7. 是否优于 DPO-v8：{'是' if better_v8 else '否'}。",
        f"8. 是否优于 DPO-v8.1：{'是' if better_v81 else '否'}。",
        f"9. 当前最佳 checkpoint：`{best}`。",
        f"10. 是否建议继续 DPO：{'是' if continue_dpo else '否'}；本轮不自动追加 steps。",
        f"11. 是否建议构造 v8.3：{'是' if report['recommend_build_v8_3'] else '否'}。",
        "12. 是否建议进入 GRPO-lite：否；Stage 10 明确禁止进入 GRPO-lite。",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "best_checkpoint": best, "stage10_success": report["stage10_success"], "recommend_enter_grpo_lite": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
