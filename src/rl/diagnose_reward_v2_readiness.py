"""Make a conservative readiness decision for future GRPO-lite reward smoke."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose reward v2 readiness.")
    parser.add_argument("--summary", default="outputs/final_report/grpo_lite_reward_v2_summary.csv")
    parser.add_argument("--by_failure", default="outputs/final_report/grpo_lite_reward_v2_by_failure_type.csv")
    parser.add_argument("--sensitivity", default="outputs/final_report/grpo_lite_reward_v2_sensitivity.csv")
    parser.add_argument("--case_study", default="outputs/final_report/grpo_lite_reward_v2_case_study.csv")
    parser.add_argument("--output_md", default="outputs/final_report/grpo_lite_reward_v2_readiness.md")
    parser.add_argument("--output_json", default="outputs/final_report/grpo_lite_reward_v2_readiness.json")
    args = parser.parse_args()
    summary = rows(args.summary)
    failures = rows(args.by_failure)
    sensitivity = rows(args.sensitivity)
    case_study = rows(args.case_study)
    model = {(row["dataset"], row["model_name"]): row for row in summary}
    r3_l = model[("lingoqa", "SFT-v3-r3")]
    v8 = model[("lingoqa", "DPO-v8")]
    v82 = model[("lingoqa", "DPO-v8.2 step-25")]
    v81 = model[("lingoqa", "DPO-v8.1 step-25")]
    base_d = model[("drivelm", "Base Qwen2.5-VL-3B")]
    r3_d = model[("drivelm", "SFT-v3-r3")]
    spatial = next(row for row in failures if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and row["failure_type"] == "spatial_relation_failure")
    object_row = next(row for row in failures if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and row["failure_type"] == "object_token_failure")
    camera = next(row for row in failures if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and row["failure_type"] == "camera_specific_failure")
    lingo_consistent = value(r3_l, "mean_total_reward") > value(v8, "mean_total_reward") and value(r3_l, "mean_total_reward") > value(v82, "mean_total_reward")
    full_selection_explained = lingo_consistent and value(r3_l, "mean_total_reward") >= value(v81, "mean_total_reward")
    drive_penalized = value(base_d, "mean_total_reward") > value(r3_d, "mean_total_reward")
    structural_penalized = all(value(row, "mean_total_reward") < value(r3_d, "mean_total_reward") for row in (spatial, object_row, camera))
    blank_penalized = value(r3_d, "mean_blank_high_f1_penalty") > value(base_d, "mean_blank_high_f1_penalty")
    refusal_safe = any(row["case_category"] == "normal_refusal_penalty_sanity" and value(row, "total_reward") < 0 for row in case_study)
    sensitivity_ok = all(row["lingoqa_r3_above_dpo_v8_2"] == "True" and row["drivelm_base_above_r3"] == "True" for row in sensitivity)
    intuitive_cases = any(row["case_category"] == "high_reward_good" for row in case_study) and any(row["case_category"] == "low_reward_bad" for row in case_study)
    ready = all([lingo_consistent, full_selection_explained, drive_penalized, structural_penalized, blank_penalized, refusal_safe, sensitivity_ok, intuitive_cases])
    report = {
        "ready_for_grpo_lite": ready,
        "checks": {
            "lingoqa_r3_above_dpo_v8_and_v8_2": lingo_consistent,
            "lingoqa_final_selection_fully_explained_including_v8_1": full_selection_explained,
            "drivelm_ood_r3_penalized_vs_base": drive_penalized,
            "camera_spatial_object_failures_penalized": structural_penalized,
            "blank_high_f1_penalized": blank_penalized,
            "normal_refusal_hacking_guard": refusal_safe,
            "lambda_sensitivity_stable": sensitivity_ok,
            "case_study_intuitive": intuitive_cases,
        },
        "lingoqa_ranking": [row["model_name"] for row in sorted([r3_l, v8, v81, v82], key=lambda row: value(row, "mean_total_reward"), reverse=True)],
        "drivelm_ranking": [row["model_name"] for row in sorted([base_d, r3_d], key=lambda row: value(row, "mean_total_reward"), reverse=True)],
        "recommend_enter_grpo_lite_gpu_smoke": ready,
        "minimum_config_if_later_ready": {
            "init_adapter": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
            "reference_adapter": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
            "group_size": 4,
            "temperature": 0.7,
            "max_new_tokens": 64,
            "max_steps": 50,
            "heldout_training_forbidden": True,
        },
        "next_step": "Refine reward harness v2 and validate structural/case-level ranking before any GRPO run." if not ready else "Request separate authorization for a 50-step GRPO-lite reward smoke only.",
    }
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# GRPO-lite Reward Harness v2 Readiness", "",
        f"最终判定：**{'可以考虑最小 GPU smoke' if ready else 'Do not train GRPO-lite yet.'}**", "",
        "## 检查结果", "", "| check | pass |", "| --- | --- |",
    ]
    lines.extend(f"| {key} | {'是' if passed else '否'} |" for key, passed in report["checks"].items())
    lines += [
        "", "## 问题回答", "",
        f"1. Reward v2 是否能解释 LingoQA 上 r3 是主模型：{'部分可以；r3 高于 DPO-v8/v8.2，但未稳定高于 v8.1' if lingo_consistent and not full_selection_explained else ('是' if full_selection_explained else '否')}。",
        f"2. 是否惩罚 DPO-v8/v8.2 case-gap 问题：{'是' if lingo_consistent else '否'}。",
        f"3. 是否惩罚 DriveLM camera/object/spatial failures：{'是' if structural_penalized else '否'}。",
        f"4. 是否识别 blank high-F1 prior answer：{'是' if blank_penalized else '否'}。",
        f"5. 是否会鼓励全部拒答：{'当前 sanity gate 未发现此风险' if refusal_safe else '存在风险'}。",
        f"6. 对 lambda 是否过于敏感：{'是，至少一组合理扰动会破坏所需排序' if not sensitivity_ok else '未观察到明显不稳定'}。",
        f"7. 是否可以作为 GRPO-lite 训练信号：{'仅可用于后续最小 smoke' if ready else '尚不可以直接训练'}。",
        "8. 仍缺少：更稳健的结构化 penalty 校准、与人工 case 顺序的一致性复核，以及避免对当前模型排名过拟合的验证。",
        f"9. 是否建议进入 GRPO-lite GPU smoke：{'是' if ready else '否'}。",
        "10. 如未来通过 gate，最小配置：r3 初始化与 frozen r3 reference，group_size=4，temperature=0.7，max_new_tokens=64，最多 50 steps，禁止使用 held-out 训练。",
        f"11. 当前下一步：{report['next_step']}",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
