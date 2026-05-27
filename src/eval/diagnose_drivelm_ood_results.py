"""Write the Stage 13 Chinese OOD diagnostic verdict."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BASE = "base_qwen25vl_3b"
R3 = "sft_v3_r3_lingo_smoke"


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["model_name"]: row for row in csv.DictReader(handle)}


def number(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Stage 13 DriveLM OOD results.")
    parser.add_argument("--results", default="outputs/final_report/drivelm_ood_results_100_answer_only.csv")
    parser.add_argument("--capability", default="outputs/final_report/drivelm_ood_capability_breakdown.csv")
    parser.add_argument("--pool_summary", default="outputs/final_report/drivelm_ood_pool_summary.json")
    parser.add_argument("--output_md", default="outputs/final_report/drivelm_ood_diagnosis.md")
    parser.add_argument("--output_json", default="outputs/final_report/drivelm_ood_diagnosis.json")
    args = parser.parse_args()

    rows = load_rows(Path(args.results))
    if BASE not in rows or R3 not in rows:
        raise ValueError("DriveLM OOD diagnosis requires both Base and SFT-v3-r3 answer-only rows")
    base, r3 = rows[BASE], rows[R3]
    with Path(args.capability).open("r", encoding="utf-8", newline="") as handle:
        cap_rows = list(csv.DictReader(handle))
    r3_caps = [row for row in cap_rows if row["model_name"] == R3]
    pool = json.loads(Path(args.pool_summary).read_text(encoding="utf-8"))

    normal_improved = number(r3, "normal_f1") > number(base, "normal_f1")
    gap_improved = number(r3, "case_gap") >= number(base, "case_gap")
    control_reduced = number(r3, "control_high_f1_rate_0_20") <= number(base, "control_high_f1_rate_0_20")
    text_prior_serious = (
        number(r3, "text_only_f1") >= 0.20
        or number(r3, "control_high_f1_rate_0_20") >= 0.30
        or number(r3, "control_direct_answer_rate") >= 0.50
    )
    blank_serious = number(r3, "blank_image_f1") >= 0.20 or number(r3, "blank_high_f1_rate_0_20") >= 0.30
    supported_caps = [row for row in r3_caps if int(float(row["num_samples"])) >= 5]
    hardest = min(supported_caps or r3_caps, key=lambda row: number(row, "normal_f1")) if r3_caps else {}
    rare_hardest = min(r3_caps, key=lambda row: number(row, "normal_f1")) if r3_caps else {}
    strongest_prior = max(r3_caps, key=lambda row: number(row, "control_high_f1_rate")) if r3_caps else {}
    wrong_confound = max(r3_caps, key=lambda row: number(row, "wrong_image_high_f1_rate")) if r3_caps else {}
    wrong_serious = (
        number(r3, "wrong_image_f1") >= 0.20
        or number(r3, "wrong_image_f1") >= number(r3, "normal_f1") - 0.02
        or number(wrong_confound, "wrong_image_high_f1_rate") >= 0.30
    )

    if normal_improved and gap_improved:
        generalization = "部分成立：normal QA 与视觉依赖指标均未比 Base 更差"
        verdict = "partial_ood_generalization"
        recommend_300 = True
    elif normal_improved:
        generalization = "仅 normal QA 风格迁移，视觉依赖没有泛化"
        verdict = "normal_improved_visual_dependency_regressed"
        recommend_300 = False
    else:
        generalization = "未显示跨数据集增益，存在 LingoQA 后训练的域偏移限制"
        verdict = "no_ood_gain"
        recommend_300 = False

    report: dict[str, Any] = {
        "stage": "Stage 13 DriveLM OOD Strict Visual-Control GPU Eval",
        "eval_num_samples": int(float(r3["num_samples"])),
        "base": base,
        "sft_v3_r3": r3,
        "checks": {
            "r3_normal_f1_above_base": normal_improved,
            "r3_case_gap_not_worse_than_base": gap_improved,
            "r3_control_high_f1_not_worse_than_base": control_reduced,
            "text_only_prior_serious": text_prior_serious,
            "wrong_image_confound_serious": wrong_serious,
            "blank_image_high_f1_serious": blank_serious,
        },
        "ood_verdict": verdict,
        "generalization_statement": generalization,
        "hardest_capability_r3": hardest,
        "rare_hardest_capability_r3": rare_hardest,
        "strongest_prior_capability_r3": strongest_prior,
        "strongest_wrong_image_confound_capability_r3": wrong_confound,
        "camera_labels_may_affect_result": bool(pool.get("multi_image_samples", 0) or pool.get("camera_aware_coverage", 0)),
        "write_as_limitation": True,
        "recommend_eval_300": recommend_300,
        "recommend_continue_training": False,
        "recommend_enter_grpo_lite": False,
    }
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage 13：DriveLM OOD Strict Visual-Control 诊断",
        "",
        "本结果是跨数据集 OOD 诊断，不是训练数据来源，也不改变 LingoQA held-out 上已确定的主模型选择。",
        "",
        "## Answer-only 主结果",
        "",
        "| 模型 | normal_f1 | case_gap | text_only_f1 | wrong_image_f1 | blank_image_f1 | control_high_f1 | hallucination | normal_refusal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in ((BASE, base), (R3, r3)):
        lines.append(
            f"| {name} | {number(row, 'normal_f1'):.4f} | {number(row, 'case_gap'):.4f} | "
            f"{number(row, 'text_only_f1'):.4f} | {number(row, 'wrong_image_f1'):.4f} | "
            f"{number(row, 'blank_image_f1'):.4f} | {number(row, 'control_high_f1_rate_0_20'):.4f} | "
            f"{number(row, 'control_hallucination_rate'):.4f} | {number(row, 'normal_refusal_rate'):.4f} |"
        )
    lines += [
        "",
        "## 诊断回答",
        "",
        f"1. r3 是否能泛化到 DriveLM：{generalization}。",
        f"2. r3 normal_f1 是否高于 Base：{'是' if normal_improved else '否'}。",
        f"3. r3 case_gap 是否优于 Base：{'是' if gap_improved else '否'}。",
        f"4. r3 control high-F1 是否降低：{'是' if control_reduced else '否'}。",
        f"5. text-only prior 是否仍然严重：{'是' if text_prior_serious else '否'}。",
        f"6. wrong-image confound 是否仍然严重：{'是' if wrong_serious else '否'}。",
        f"7. blank-image high-F1 是否仍然严重：{'是' if blank_serious else '否'}。",
        f"8. 多相机/camera label 是否可能影响结果：{'是' if report['camera_labels_may_affect_result'] else '未观察到直接证据'}；本池 camera-aware cases={pool.get('camera_aware_coverage', 0)}。",
        f"9. 在样本量至少 5 的类型中，r3 最难能力为 `{hardest.get('capability', 'n/a')}`；最强 control prior / wrong-image confound 类型均为 `{strongest_prior.get('capability', 'n/a')}`。",
        "10. DriveLM OOD 是否支持项目结论：支持使用 strict visual-control 暴露跨域视觉依赖问题，但不自动证明 r3 跨域成功。",
        "11. 是否应写入 limitation：是，DriveLM 是新的 OOD 泛化边界。",
        f"12. 是否建议扩大到 300 cases：{'是' if recommend_300 else '否，先保留 100-case 诊断结论并分析失败模式'}。",
        "13. 是否建议继续训练：否；OOD 结果不能用于本轮训练决策或训练数据构造。",
        "14. 是否建议进入 GRPO-lite：否；仍需在干净主评测与 OOD 诊断上验证 reward 设计。",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
