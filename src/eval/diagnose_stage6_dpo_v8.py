"""Generate a Chinese diagnostic report for DPO-v8 held-out smoke results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

R3 = "sft_v3_r3_lingo_smoke"
DPO8 = "dpo_v8_lingo_smoke"
SFT2 = "sft_v2"
DPO7 = "dpo_v7"


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def answer_row(rows: list[dict[str, str]], model: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("model_name") == model and row.get("score_mode") == "answer_only"), None)


def value(row: dict[str, str] | None, key: str) -> float:
    return float(row.get(key, 0) or 0) if row else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Stage 6 DPO-v8 held-out results.")
    parser.add_argument("--results", default="outputs/final_report/stage6_dpo_v8_heldout_main_results.csv")
    parser.add_argument("--train_summary", default="outputs/train_logs/dpo_v8_lingo_smoke/train_summary.json")
    parser.add_argument("--output_json", default="outputs/final_report/stage6_dpo_v8_diagnosis.json")
    parser.add_argument("--output_md", default="outputs/final_report/stage6_dpo_v8_diagnosis.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.results))
    summary_path = Path(args.train_summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    r3, dpo8, sft2, dpo7 = (answer_row(rows, model) for model in (R3, DPO8, SFT2, DPO7))
    available = all((r3, dpo8, sft2, dpo7))
    training_success = summary.get("success") is True and not summary.get("dry_run", False)
    frozen_r3_reference = (
        summary.get("reference_free") is False
        and summary.get("init_adapter") == summary.get("reference_adapter")
        and "sft_v3_r3_lingo_smoke" in str(summary.get("reference_adapter", ""))
    )
    report: dict[str, Any] = {
        "training_success": training_success,
        "frozen_r3_reference": frozen_r3_reference,
        "heldout_results_available": available,
        "recommend_enter_grpo_lite": False,
    }
    lines = ["# Stage 6：DPO-v8 GPU Smoke 诊断", ""]
    if not available:
        report.update({"dpo_success": False, "recommend_continue_dpo": False, "recommend_model_mined_v8_1": False})
        lines += [
            "held-out 结果尚不完整，当前不能判断 DPO-v8 是否改善 r3。",
            "",
            "- 是否建议进入 GRPO-lite：否。",
        ]
    else:
        normal_preserved = value(dpo8, "normal_f1") >= value(r3, "normal_f1") - 0.03
        hallucination_reduced = value(dpo8, "control_hallucination_rate") < value(r3, "control_hallucination_rate")
        case_gap_improved = value(dpo8, "case_gap") >= value(r3, "case_gap")
        refusal_ok = value(dpo8, "normal_refusal_rate") <= 0.02
        length_abnormal = value(dpo8, "avg_answer_length") > max(value(r3, "avg_answer_length") * 1.25, value(r3, "avg_answer_length") + 8)
        stronger_than_prior = value(dpo8, "normal_f1") >= max(value(sft2, "normal_f1"), value(dpo7, "normal_f1"))
        success = training_success and frozen_r3_reference and normal_preserved and hallucination_reduced and case_gap_improved and refusal_ok and not length_abnormal
        report.update({
            "checks": {
                "normal_f1_preserved_within_0_03": normal_preserved,
                "control_hallucination_reduced": hallucination_reduced,
                "case_gap_not_worse_than_r3": case_gap_improved,
                "normal_refusal_at_most_0_02": refusal_ok,
                "answer_length_abnormal": length_abnormal,
                "normal_f1_ge_sft2_and_dpo7": stronger_than_prior,
            },
            "r3": r3,
            "dpo_v8": dpo8,
            "dpo_success": success,
            "recommend_continue_dpo": success,
            "recommend_model_mined_v8_1": not hallucination_reduced or not success,
        })
        lines += [
            f"- DPO-v8 是否成功训练：{'是' if training_success else '否'}。",
            f"- 是否使用 frozen r3 reference：{'是' if frozen_r3_reference else '否'}。",
            "",
            "## Answer-only Held-out 指标",
            "",
            f"- r3 normal_f1: {value(r3, 'normal_f1'):.4f}",
            f"- DPO-v8 normal_f1: {value(dpo8, 'normal_f1'):.4f}",
            f"- r3 case_gap: {value(r3, 'case_gap'):.4f}",
            f"- DPO-v8 case_gap: {value(dpo8, 'case_gap'):.4f}",
            f"- r3 control_hallucination: {value(r3, 'control_hallucination_rate'):.4f}",
            f"- DPO-v8 control_hallucination: {value(dpo8, 'control_hallucination_rate'):.4f}",
            f"- DPO-v8 normal_refusal: {value(dpo8, 'normal_refusal_rate'):.4f}",
            f"- r3 avg_answer_length: {value(r3, 'avg_answer_length'):.4f}",
            f"- DPO-v8 avg_answer_length: {value(dpo8, 'avg_answer_length'):.4f}",
            "",
            "## 判断",
            "",
            f"- normal QA 是否保住 r3：{'是' if normal_preserved else '否'}。",
            f"- normal QA 是否仍不低于 SFT-v2 / DPO-v7：{'是' if stronger_than_prior else '否'}。",
            f"- control hallucination 是否低于 r3：{'是' if hallucination_reduced else '否'}。",
            f"- case_gap 是否优于或等于 r3：{'是' if case_gap_improved else '否'}。",
            f"- normal refusal 是否异常上升：{'否' if refusal_ok else '是'}。",
            f"- answer length 是否异常变长：{'是' if length_abnormal else '否'}。",
            f"- DPO-v8 是否比 r3 更适合作为当前候选：{'是' if success else '否'}。",
            "",
            "## 决策",
            "",
        ]
        if success:
            lines += [
                "- 当前 smoke 达到预设成功条件；可以把 DPO-v8 作为候选继续复核，但不应直接盲目增加 steps。",
                "- 后续优先扩展真实 model-mined preference_v8.1 或增加 held-out 样本后再次验证。",
            ]
        elif not normal_preserved:
            lines += [
                "- 不建议继续加 DPO steps；normal F1 相比 r3 下降超过容忍范围，应回退 r3 并增加 normal anchor。",
            ]
        elif not hallucination_reduced:
            lines += [
                "- 不建议继续加 DPO steps；rule-based pairs 未有效降低 hallucination，应构造 model-mined preference_v8.1 hard negatives。",
            ]
        else:
            lines += [
                "- 当前未满足完整成功条件，不建议直接加步数；应审查 case gallery 并调整 preference 设计。",
            ]
        lines += [
            "- 是否建议进入 GRPO-lite：否。只有 DPO 稳定且 reward harness 通过后仍存在问题，才考虑 GRPO-lite。",
        ]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"dpo_success": report.get("dpo_success", False), "recommend_enter_grpo_lite": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
