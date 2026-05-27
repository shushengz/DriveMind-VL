"""Generate the CPU-only Stage 9 Preference-v8.2 readiness diagnosis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing Stage 9 input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Preference-v8.2 data readiness.")
    parser.add_argument("--stats", default="data/train/preference_v8_2/preference_v8_2_stats.json")
    parser.add_argument("--audit", default="outputs/data_audit/preference_v8_2_audit.json")
    parser.add_argument("--design", default="outputs/final_report/preference_v8_2_design.md")
    parser.add_argument("--attribution", default="outputs/final_report/stage8_5_error_attribution.json")
    parser.add_argument("--output_md", default="outputs/final_report/stage9_preference_v8_2_diagnosis.md")
    parser.add_argument("--output_json", default="outputs/final_report/stage9_preference_v8_2_diagnosis.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    stats = read_json(Path(args.stats))
    audit = read_json(Path(args.audit))
    attribution = read_json(Path(args.attribution))
    design_exists = Path(args.design).exists()
    train_ready = bool(audit.get("train_ready"))
    blank_ok = int(audit.get("blank_high_f1_pair_count", 0)) >= 30
    direct_ok = int(audit.get("control_direct_answer_pair_count", 0)) >= 50
    normal_improved = float(audit.get("normal_pair_ratio", 0.0)) >= 0.35
    control_reduced = float(audit.get("control_pair_ratio", 1.0)) <= 0.60
    clean = (
        int(audit.get("heldout_leakage_count", -1)) == 0
        and int(audit.get("reason_field_count", -1)) == 0
        and int(audit.get("metadata_leak_count", -1)) == 0
    )
    recommend_dpo = train_ready and blank_ok and direct_ok and normal_improved and control_reduced and clean
    report = {
        "cpu_only": True,
        "addresses_stage8_5_failure_mode": True,
        "primary_regression_setting": attribution.get("primary_case_gap_regression_setting", "blank_image"),
        "blank_high_f1_pairs_sufficient": blank_ok,
        "control_direct_answer_pairs_sufficient": direct_ok,
        "normal_anchor_increased_vs_v8_1": normal_improved,
        "control_ratio_reduced_vs_v8_1": control_reduced,
        "heldout_leakage_count": audit.get("heldout_leakage_count", 0),
        "reason_field_count": audit.get("reason_field_count", 0),
        "metadata_leak_count": audit.get("metadata_leak_count", 0),
        "train_ready": train_ready,
        "recommend_enter_dpo_v8_2_gpu_smoke": recommend_dpo,
        "recommend_enter_grpo_lite": False,
        "pair_type_counts": audit.get("pair_type_counts", {}),
        "pair_type_ratios": audit.get("pair_type_ratios", {}),
        "warnings": stats.get("warnings", []) + audit.get("warnings", []),
        "design_document_present": design_exists,
    }
    lines = [
        "# Stage 9: Preference-v8.2 数据诊断",
        "",
        "本报告仅基于离线数据构造与审计结果，不包含模型训练或推理结论。",
        "",
        "## 结论",
        "",
        f"1. v8.2 是否针对 Stage 8.5 的失败模式：是。其目标是压制 `{report['primary_regression_setting']}` 高 F1 先验回答与 control direct answers。",
        f"2. blank_high_f1 pair 是否足够：{'是' if blank_ok else '否'}（{audit.get('blank_high_f1_pair_count', 0)} 条）。",
        f"3. control_direct_answer pair 是否足够：{'是' if direct_ok else '否'}（{audit.get('control_direct_answer_pair_count', 0)} 条）。",
        f"4. normal anchor 是否提高到安全区间：{'是' if normal_improved else '否'}（normal-related {float(audit.get('normal_pair_ratio', 0)):.2%}）。",
        f"5. control ratio 是否低于 v8.1 的 67%：{'是' if control_reduced else '否'}（control-related {float(audit.get('control_pair_ratio', 0)):.2%}）。",
        f"6. 是否存在 held-out leakage：{'否' if audit.get('heldout_leakage_count', 0) == 0 else '是'}。",
        f"7. 是否存在 reason / metadata 泄漏：{'否' if clean else '是'}。",
        f"8. 数据是否 train_ready：{'是' if train_ready else '否'}。",
        f"9. 是否建议进入 DPO-v8.2 GPU smoke：{'是，仅建议 25-step smoke' if recommend_dpo else '否'}。",
        "10. 是否建议进入 GRPO-lite：否。应先验证 case-gap-aware DPO 是否真正修复 held-out control overlap。",
        "",
        "## Pair 分布",
        "",
    ]
    for pair_type, count in sorted(audit.get("pair_type_counts", {}).items()):
        ratio = float(audit.get("pair_type_ratios", {}).get(pair_type, 0.0))
        lines.append(f"- `{pair_type}`: {count} ({ratio:.2%})")
    if report["warnings"]:
        lines += ["", "## Warnings"] + [f"- {warning}" for warning in report["warnings"]]
    lines += [
        "",
        "## 下一步",
        "",
        "若另行授权 GPU smoke，应使用 frozen r3 reference，从 r3 初始化，仅运行 DPO-v8.2 25 steps 并继续在无泄漏 held-out 100 IDs 上评估；不要自动增加步数，也不要自动进入 GRPO-lite。",
    ]
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not train_ready:
        raise SystemExit("Preference-v8.2 is not train-ready; DPO-v8.2 GPU smoke remains blocked")


if __name__ == "__main__":
    main()
