"""Strict offline readiness gate for Reward v2.2 after human review integration."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def read(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def main() -> None:
    summary = read("outputs/final_report/grpo_lite_reward_v2_2_summary.csv")
    sensitivity = read("outputs/final_report/grpo_lite_reward_v2_2_sensitivity.csv")
    pairwise = read("outputs/final_report/grpo_lite_reward_v2_2_pairwise.csv")
    impact = read("outputs/final_report/grpo_lite_reward_v2_2_manual_review_impact.csv")
    manual = json.loads(Path("outputs/final_report/reward_manual_review_summary.json").read_text(encoding="utf-8"))
    models = {(row["dataset"], row["model_name"]): row for row in summary}
    lingo = sorted([row for row in summary if row["dataset"] == "lingoqa"], key=lambda row: value(row, "mean_total_reward"), reverse=True)
    drive = sorted([row for row in summary if row["dataset"] == "drivelm"], key=lambda row: value(row, "mean_total_reward"), reverse=True)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pairwise:
        groups[row["pair_type"]].append(row)
    accuracy = lambda kind: sum(row["correct"] == "True" for row in groups.get(kind, [])) / len(groups[kind]) if groups.get(kind) else 0.0
    overall = sum(row["correct"] == "True" for row in pairwise) / len(pairwise) if pairwise else 0.0
    under = [row for row in impact if row["human_label"] == "reward_under_penalized"]
    under_ratio = sum(row["under_penalized_improved"] == "True" for row in under) / len(under) if under else 0.0
    over = [row for row in impact if row["human_label"] == "reward_over_penalized"]
    over_harmed = sum(row["over_penalized_further_harmed"] == "True" for row in over)
    default_drive_r3 = models[("drivelm", "SFT-v3-r3")]
    patch_status = {
        "object_token_invariant_penalty_active": value(default_drive_r3, "mean_object_token_invariant_penalty") > 0,
        "normal_object_category_mismatch_penalty_active": any(value(row, "mean_normal_object_category_mismatch_penalty") > 0 for row in summary),
        "invalid_generic_answer_penalty_active": any(value(row, "mean_invalid_generic_answer_penalty") > 0 for row in summary),
        "control_same_as_normal_penalty_active": any(value(row, "mean_control_same_as_normal_penalty") > 0 for row in summary),
    }
    checks = {
        "manual_review_file_read": bool(manual.get("manual_review_file_found")),
        "manual_review_has_no_blank_labels": manual.get("blocking_missing_label_count", 1) == 0,
        "sensitivity_basic_stable": (
            sum(row["drivelm_base_gt_r3"] == "True" for row in sensitivity) >= 8
            and sum(row["r3_ge_dpo_v8"] == "True" for row in sensitivity) >= 8
            and sum(row["r3_ge_dpo_v8_2"] == "True" for row in sensitivity) >= 8
        ),
        "overall_pairwise_accuracy_ge_0_95": overall >= 0.95,
        "blank_pair_accuracy_ge_0_90": accuracy("blank_low_vs_blank_high") >= 0.90,
        "refusal_pair_accuracy_ge_0_95": accuracy("normal_correct_vs_normal_refusal") >= 0.95,
        "spatial_pair_accuracy_ge_0_90": accuracy("drive_spatial_good_vs_spatial_bad") >= 0.90,
        "object_invariant_pair_accuracy_ge_0_80": accuracy("object_invariant_bad_vs_object_grounded_good") >= 0.80,
        "invalid_generic_pair_accuracy_ge_0_95": accuracy("invalid_generic_bad_vs_valid_answer_good") >= 0.95,
        "control_same_pair_accuracy_ge_0_80": accuracy("control_same_as_normal_bad_vs_control_caution_good") >= 0.80,
        "manual_under_penalized_majority_improved": under_ratio >= 0.70,
        "manual_over_penalized_not_further_harmed": over_harmed == 0,
        "drivelm_base_above_r3_default": drive[0]["model_name"] == "Base Qwen2.5-VL-3B",
        "r3_not_below_v8_or_v8_2_default": (
            value(models[("lingoqa", "SFT-v3-r3")], "mean_total_reward") >= value(models[("lingoqa", "DPO-v8")], "mean_total_reward")
            and value(models[("lingoqa", "SFT-v3-r3")], "mean_total_reward") >= value(models[("lingoqa", "DPO-v8.2 step-25")], "mean_total_reward")
        ),
        "normal_refusal_low_all_sensitivity": all(row["normal_refusal_low"] == "True" for row in sensitivity),
        "new_patches_trigger_on_drivelm_r3": (
            value(default_drive_r3, "mean_object_token_invariant_penalty") > 0
            and value(default_drive_r3, "mean_control_same_as_normal_penalty") > 0
        ),
    }
    ready = all(checks.values())
    report = {
        "reward_v2_2_ready_for_grpo_lite_smoke": ready,
        "readiness_scope": "offline_gate_passed_after_manual_review_integration" if ready else "offline_gate_failed_after_manual_review_integration",
        "checks": checks,
        "fail_reasons": [key for key, ok in checks.items() if not ok],
        "human_label_counts": manual.get("human_label_counts", {}),
        "lingoqa_ranking": [row["model_name"] for row in lingo],
        "drivelm_ranking": [row["model_name"] for row in drive],
        "overall_pairwise_accuracy": overall,
        "pairwise_by_type": {key: {"count": len(rows), "accuracy": accuracy(key)} for key, rows in groups.items()},
        "sensitivity_configs": len(sensitivity),
        "manual_under_penalized_total": len(under),
        "manual_under_penalized_improved": sum(row["under_penalized_improved"] == "True" for row in under),
        "manual_under_penalized_improved_rate": under_ratio,
        "patch_status": patch_status,
        "recommend_enter_grpo_lite_gpu_smoke": ready,
        "recommendation": "Only allow a strictly controlled 50-step GRPO-lite smoke after explicit approval." if ready else "Do not train GRPO-lite yet.",
        "minimum_grpo_config_if_authorized": {
            "init_adapter": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
            "reference_adapter": "checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/",
            "reference_frozen": True, "group_size": 4, "temperature": 0.7, "max_new_tokens": 64, "max_steps": 50,
            "heldout_training_forbidden": True, "drivelm_ood_training_forbidden": True,
        },
    }
    Path("outputs/final_report/grpo_lite_reward_v2_2_readiness.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Reward v2.2 Readiness", "", f"结论：**{report['recommendation']}**", "",
             "## Gate Results", "", "| gate | pass |", "| --- | --- |"]
    lines.extend(f"| {key} | {'是' if ok else '否'} |" for key, ok in checks.items())
    lines += ["", "## 核心指标", "",
              f"- LingoQA ranking：`{' > '.join(report['lingoqa_ranking'])}`。",
              f"- DriveLM ranking：`{' > '.join(report['drivelm_ranking'])}`。",
              f"- Overall pairwise accuracy：{overall:.4f}。",
              f"- Object-invariant pair accuracy：{accuracy('object_invariant_bad_vs_object_grounded_good'):.4f}。",
              f"- Control-same-as-normal pair accuracy：{accuracy('control_same_as_normal_bad_vs_control_caution_good'):.4f}。",
              f"- Normal-refusal pair accuracy：{accuracy('normal_correct_vs_normal_refusal'):.4f}。",
              f"- 人工标记为漏罚的样本改善：{report['manual_under_penalized_improved']}/{len(under)}。",
              f"- 新增 penalty 是否实际触发：{json.dumps(patch_status, ensure_ascii=False)}。",
              "", "v2.2 只完成离线 reward 审计；即使通过，也不能自动触发训练或使用 held-out/OOD 样本训练。", ""]
    Path("outputs/final_report/grpo_lite_reward_v2_2_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
