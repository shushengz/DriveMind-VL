"""Diagnose DPO-v8.1 checkpoints on held-out strict visual-control results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

R3 = "sft_v3_r3_lingo_smoke"
DPO8 = "dpo_v8_lingo_smoke"
STEP25 = "dpo_v8_1_lingo_smoke_step25"
STEP50 = "dpo_v8_1_lingo_smoke_step50"


def answer_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["model_name"]: row for row in csv.DictReader(handle) if row.get("score_mode") == "answer_only"}


def val(row: dict[str, str] | None, field: str) -> float:
    return float((row or {}).get(field, 0) or 0)


def assess(candidate: dict[str, str], r3: dict[str, str], dpo8: dict[str, str] | None) -> dict[str, Any]:
    normal_ok = val(candidate, "normal_f1") >= val(r3, "normal_f1") - 0.03
    hallucination_better = val(candidate, "control_hallucination_rate") < val(r3, "control_hallucination_rate")
    gap_better = val(candidate, "case_gap") >= val(r3, "case_gap")
    refusal_ok = val(candidate, "normal_refusal_rate") <= 0.02
    length_bad = val(candidate, "avg_answer_length") > max(val(r3, "avg_answer_length") * 1.25, val(r3, "avg_answer_length") + 8)
    success = normal_ok and hallucination_better and gap_better and refusal_ok and not length_bad
    excellent = (
        val(candidate, "normal_f1") >= val(r3, "normal_f1")
        and dpo8 is not None
        and val(candidate, "control_hallucination_rate") <= val(dpo8, "control_hallucination_rate")
        and val(candidate, "case_gap") > 0
        and val(candidate, "normal_refusal_rate") == 0
    )
    return {
        "normal_preserved": normal_ok,
        "control_hallucination_reduced": hallucination_better,
        "case_gap_improved": gap_better,
        "normal_refusal_ok": refusal_ok,
        "answer_length_abnormal": length_bad,
        "success": success,
        "excellent": excellent,
    }


def choose_best(rows: dict[str, dict[str, str]], checks: dict[str, dict[str, Any]]) -> str:
    candidates = [name for name in (STEP25, STEP50) if name in rows]
    return max(
        candidates,
        key=lambda name: (
            checks[name]["success"],
            checks[name]["excellent"],
            val(rows[name], "case_gap"),
            val(rows[name], "normal_f1"),
            -val(rows[name], "control_hallucination_rate"),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stage 8 DPO-v8.1 held-out results.")
    parser.add_argument("--results", default="outputs/final_report/stage8_dpo_v8_1_heldout_main_results.csv")
    parser.add_argument("--train_summary", default="outputs/train_logs/dpo_v8_1_lingo_smoke/train_summary.json")
    parser.add_argument("--output_json", default="outputs/final_report/stage8_dpo_v8_1_diagnosis.json")
    parser.add_argument("--output_md", default="outputs/final_report/stage8_dpo_v8_1_diagnosis.md")
    args = parser.parse_args()
    rows = answer_rows(Path(args.results))
    summary = json.loads(Path(args.train_summary).read_text(encoding="utf-8")) if Path(args.train_summary).exists() else {}
    audit_path = Path("outputs/data_audit/preference_v8_1_audit.json")
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    missing = [name for name in (R3, DPO8, STEP25, STEP50) if name not in rows]
    frozen_r3 = summary.get("reference_free") is False and summary.get("init_adapter") == summary.get("reference_adapter") and "sft_v3_r3_lingo_smoke" in str(summary.get("reference_adapter", ""))
    report: dict[str, Any] = {
        "training_success": summary.get("success") is True and not summary.get("dry_run", False),
        "uses_model_mined_preference": audit.get("uses_model_predictions") is True,
        "frozen_r3_reference": frozen_r3,
        "missing_results": missing,
        "recommend_enter_grpo_lite": False,
    }
    lines = ["# Stage 8：DPO-v8.1 Model-mined GPU Smoke 诊断", ""]
    if missing:
        report.update({"stage8_success": False, "best_checkpoint": None, "recommend_continue_dpo": False, "recommend_build_v8_2": False})
        lines += ["held-out 结果尚不完整，不能对 DPO-v8.1 作性能判断。", "", "- 是否建议进入 GRPO-lite：否。"]
    else:
        checks = {name: assess(rows[name], rows[R3], rows[DPO8]) for name in (STEP25, STEP50)}
        best = choose_best(rows, checks)
        candidate = rows[best]
        better_than_dpo8 = val(candidate, "control_hallucination_rate") <= val(rows[DPO8], "control_hallucination_rate") and val(candidate, "case_gap") >= val(rows[DPO8], "case_gap")
        report.update({
            "checks": checks,
            "best_checkpoint": best,
            "stage8_success": checks[best]["success"],
            "better_than_rule_based_dpo_v8": better_than_dpo8,
            "recommend_continue_dpo": checks[best]["success"] and best == STEP50,
            "recommend_build_v8_2": not checks[best]["success"],
            "r3": rows[R3],
            "dpo_v8": rows[DPO8],
            "step25": rows[STEP25],
            "step50": rows[STEP50],
        })
        lines += [
            f"- DPO-v8.1 是否成功训练：{'是' if report['training_success'] else '否'}。",
            f"- 是否使用 model-mined preference：{'是' if report['uses_model_mined_preference'] else '否'}。",
            f"- 是否使用 frozen r3 reference：{'是' if frozen_r3 else '否'}。",
            "",
            "## Answer-only Held-out 指标",
            "",
            "| 模型 | Normal F1 | Case Gap | Control Hallucination | Normal Refusal | Avg Length |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in (R3, DPO8, STEP25, STEP50):
            row = rows[name]
            lines.append(f"| {name} | {val(row, 'normal_f1'):.4f} | {val(row, 'case_gap'):.4f} | {val(row, 'control_hallucination_rate'):.4f} | {val(row, 'normal_refusal_rate'):.4f} | {val(row, 'avg_answer_length'):.4f} |")
        lines += [
            "",
            "## 判断",
            "",
            f"- step-25 和 step-50 哪个更好：`{best}`。",
            f"- best checkpoint 是否保住 r3 normal F1：{'是' if checks[best]['normal_preserved'] else '否'}。",
            f"- best checkpoint 是否降低 r3 control hallucination：{'是' if checks[best]['control_hallucination_reduced'] else '否'}。",
            f"- best checkpoint 是否改善 r3 case_gap：{'是' if checks[best]['case_gap_improved'] else '否'}。",
            f"- best checkpoint 是否优于 rule-based DPO-v8：{'是' if better_than_dpo8 else '否'}。",
            f"- 是否出现 normal refusal 风险：{'否' if checks[best]['normal_refusal_ok'] else '是'}。",
            f"- answer length 是否异常：{'是' if checks[best]['answer_length_abnormal'] else '否'}。",
            "",
            "## 决策",
            "",
        ]
        if checks[best]["success"] and best == STEP25:
            lines.append("- step-25 已提供更稳的折中，选择 step-25；不建议继续增加 DPO steps，以免过度校准。")
        elif checks[best]["success"]:
            lines.append("- step-50 达到成功条件；可以扩大 held-out 复核，但不能自动追加训练。")
        else:
            lines.append("- 当前未满足 DPO-v8.1 成功条件；不建议继续加 steps，应分析失败类型并考虑 v8.2 数据修正。")
        lines.append("- 是否建议进入 GRPO-lite：否。本阶段不进入 GRPO，且须先得到稳定的 DPO held-out 证据。")
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_checkpoint": report.get("best_checkpoint"), "stage8_success": report.get("stage8_success", False), "recommend_enter_grpo_lite": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
