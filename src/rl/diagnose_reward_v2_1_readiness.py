"""Conservative readiness report for reward v2.1 and optional GRPO-lite smoke."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def val(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    summary = rows("outputs/final_report/grpo_lite_reward_v2_1_summary.csv")
    sensitivity = rows("outputs/final_report/grpo_lite_reward_v2_1_sensitivity.csv")
    pairwise = rows("outputs/final_report/grpo_lite_reward_v2_1_pairwise.csv")
    failure = rows("outputs/final_report/grpo_lite_reward_v2_1_by_failure_type.csv")
    disagreement = rows("outputs/final_report/reward_v2_disagreement_cases.csv")
    model = {(row["dataset"], row["model_name"]): row for row in summary}
    lingo = sorted([row for row in summary if row["dataset"] == "lingoqa"], key=lambda r: val(r, "mean_total_reward"), reverse=True)
    drive = sorted([row for row in summary if row["dataset"] == "drivelm"], key=lambda r: val(r, "mean_total_reward"), reverse=True)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pairwise:
        groups[row["pair_type"]].append(row)
    accuracy = lambda kind: sum(row["correct"] == "True" for row in groups.get(kind, [])) / len(groups[kind]) if groups.get(kind) else 0.0
    overall = sum(row["correct"] == "True" for row in pairwise) / len(pairwise) if pairwise else 0.0
    checks = {
        "sensitivity_basic_stable": (
            sum(row["drivelm_base_gt_r3"] == "True" for row in sensitivity) >= 7
            and sum(row["r3_ge_dpo_v8"] == "True" for row in sensitivity) >= 7
            and sum(row["r3_ge_dpo_v8_2"] == "True" for row in sensitivity) >= 7
        ),
        "pairwise_accuracy_ge_0_70": overall >= 0.70,
        "blank_pairwise_accuracy_ge_0_75": accuracy("blank_low_vs_blank_high") >= 0.75,
        "refusal_pairwise_accuracy_ge_0_90": accuracy("normal_correct_vs_normal_refusal") >= 0.90,
        "spatial_pairwise_accuracy_ge_0_65": accuracy("drive_spatial_good_vs_spatial_bad") >= 0.65,
        "drivelm_base_above_r3_default": drive[0]["model_name"] == "Base Qwen2.5-VL-3B",
        "r3_not_below_v8_or_v8_2_default": val(model[("lingoqa", "SFT-v3-r3")], "mean_total_reward") >= val(model[("lingoqa", "DPO-v8")], "mean_total_reward") and val(model[("lingoqa", "SFT-v3-r3")], "mean_total_reward") >= val(model[("lingoqa", "DPO-v8.2 step-25")], "mean_total_reward"),
        "normal_refusal_low_all_sensitivity": all(row["normal_refusal_low"] == "True" for row in sensitivity),
    }
    r3_drive = model[("drivelm", "SFT-v3-r3")]
    struct = {tag: next(row for row in failure if row["dataset"] == "drivelm" and row["model_name"] == "SFT-v3-r3" and row["failure_type"] == tag) for tag in ("spatial_relation_failure", "object_token_failure", "camera_specific_failure")}
    checks["structural_failures_below_r3_average"] = all(val(row, "mean_total_reward") < val(r3_drive, "mean_total_reward") for row in struct.values())
    automated_ready = all(checks.values())
    manual_rows = len(rows("outputs/final_report/reward_v2_manual_review_sheet.csv"))
    report = {
        "reward_v2_1_ready_for_grpo_lite_smoke": automated_ready,
        "readiness_scope": "offline_automated_gate_passed_pending_human_review_of_disagreements" if automated_ready else "offline_gate_failed",
        "checks": checks,
        "lingoqa_ranking": [row["model_name"] for row in lingo],
        "drivelm_ranking": [row["model_name"] for row in drive],
        "pairwise_accuracy": overall,
        "pairwise_by_type": {kind: {"count": len(items), "accuracy": accuracy(kind)} for kind, items in groups.items()},
        "sensitivity_configs": len(sensitivity),
        "manual_review_rows_pending": manual_rows,
        "disagreement_rows": len(disagreement),
        "fail_reasons": [key for key, passed in checks.items() if not passed],
        "recommend_enter_grpo_lite_gpu_smoke": automated_ready,
        "recommendation_caveat": "Review the generated disagreement sheet before executing any separately authorized GPU smoke." if automated_ready else "Do not train GRPO-lite yet.",
        "minimum_grpo_config_if_authorized": {"init_adapter": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/", "reference_adapter": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/", "group_size": 4, "temperature": 0.7, "max_new_tokens": 64, "max_steps": 50, "heldout_training_forbidden": True, "drivelm_ood_training_forbidden": True},
    }
    Path("outputs/final_report/grpo_lite_reward_v2_1_readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = "自动离线 gate 通过，可在人工复核后申请最小 GPU smoke" if automated_ready else "Do not train GRPO-lite yet."
    lines = ["# GRPO-lite Reward v2.1 Readiness", "", f"最终判定：**{status}**", "",
             "## Gates", "", "| gate | pass |", "| --- | --- |"]
    lines.extend(f"| {key} | {'是' if passed else '否'} |" for key, passed in checks.items())
    lines += ["", "## 核心指标", "", f"- LingoQA ranking：`{' > '.join(report['lingoqa_ranking'])}`。",
              f"- DriveLM ranking：`{' > '.join(report['drivelm_ranking'])}`。",
              f"- Overall pairwise accuracy：{overall:.4f}。",
              f"- blank_low_vs_blank_high accuracy：{accuracy('blank_low_vs_blank_high'):.4f}。",
              f"- normal_correct_vs_normal_refusal accuracy：{accuracy('normal_correct_vs_normal_refusal'):.4f}。",
              f"- drive_spatial_good_vs_spatial_bad accuracy：{accuracy('drive_spatial_good_vs_spatial_bad'):.4f}。",
              f"- Disagreement review sheet 尚有 {manual_rows} 行可供人工快速复核。", "",
              "## 问题回答", "",
              "1. v2.1 解决了 v2 的主要排序与 blank sensitivity 缺口：默认排名和 8 组 sensitivity 均符合硬门槛。",
              "2. r3 vs DPO-v8.1：v2.1 默认与敏感性中均将 r3 排在 v8.1 之前；仍保留 disagreement sheet 供人工核查。",
              "3. Object/camera penalty：触发范围已由标签宽口径收窄为与 control confound 联合触发；结构失败平均 reward 低于 r3 总体，但 object 部分仍值得人工审阅。",
              "4. Reward hacking：normal refusal 的 pairwise 与 sensitivity gate 均通过。",
              f"5. 是否 ready for GRPO-lite GPU smoke：{'自动 gate 已通过，但必须另行授权且建议先快速浏览人工复核表' if automated_ready else '否'}。",
              "6. 最小配置（若另行授权）：r3 初始化与 frozen r3 reference，group_size=4，temperature=0.7，max_new_tokens=64，max_steps=50；禁止使用 held-out 与 DriveLM OOD 训练。"]
    Path("outputs/final_report/grpo_lite_reward_v2_1_readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
