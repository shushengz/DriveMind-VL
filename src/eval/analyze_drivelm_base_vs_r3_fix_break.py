"""Compare cases improved or damaged by r3 on DriveLM OOD."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.drivelm_ood_attribution_utils import CONTROL_SETTINGS, PREDICTION_ROOT, load_cases, taxonomy_row, write_csv


def control_max(case: dict[str, Any], model: str) -> float:
    return max(float(case[model][setting]["f1"]) for setting in CONTROL_SETTINGS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Base-to-r3 DriveLM fixes and breaks.")
    parser.add_argument("--prediction_root", default=PREDICTION_ROOT.as_posix())
    parser.add_argument("--fix_csv", default="outputs/final_report/drivelm_base_vs_r3_fix_cases.csv")
    parser.add_argument("--break_csv", default="outputs/final_report/drivelm_base_vs_r3_break_cases.csv")
    parser.add_argument("--report", default="outputs/final_report/drivelm_base_vs_r3_fix_break_report.md")
    args = parser.parse_args()
    fixes, breaks = [], []
    for case in load_cases(Path(args.prediction_root)):
        row = taxonomy_row(case)
        base_control, r3_control = control_max(case, "base"), control_max(case, "r3")
        normal_fix = case["delta_normal_f1"] > 0.15 and case["delta_case_gap"] >= -0.05
        control_fix = base_control - r3_control > 0.15
        normal_break = float(case["base"]["normal"]["f1"]) >= 0.25 and case["delta_normal_f1"] < -0.10
        control_break = r3_control - base_control > 0.15
        new_blank = float(case["r3"]["blank_image"]["f1"]) >= 0.20 and float(case["base"]["blank_image"]["f1"]) < 0.20
        detail = {
            **row, "base_control_max_f1": base_control, "r3_control_max_f1": r3_control,
            "fix_reasons": "|".join(reason for reason, yes in (("normal_gain_without_gap_damage", normal_fix), ("control_overlap_reduced", control_fix)) if yes),
            "break_reasons": "|".join(reason for reason, yes in (("normal_regression", normal_break), ("control_overlap_increased", control_break), ("new_blank_high_f1", new_blank)) if yes),
        }
        if normal_fix or control_fix:
            fixes.append(detail)
        if normal_break or control_break or new_blank:
            breaks.append(detail)
    fields = list((fixes or breaks or [{}])[0]) if (fixes or breaks) else ["id"]
    write_csv(Path(args.fix_csv), fixes, fields)
    write_csv(Path(args.break_csv), breaks, fields)
    fix_caps = Counter(row["capability"] for row in fixes)
    break_caps = Counter(row["capability"] for row in breaks)
    overlap = len({row["id"] for row in fixes} & {row["id"] for row in breaks})
    lines = [
        "# DriveLM Base vs r3 Fix/Break Analysis", "",
        f"- Fix cases：{len(fixes)}。",
        f"- Break cases：{len(breaks)}。",
        f"- 同时满足不同维度 fix 与 break 的 cases：{overlap}（说明 normal 与 visual-dependency 指标可能方向相反）。",
        "", "## Capability Concentration", "",
        "| category | fix count | break count |", "| --- | ---: | ---: |",
    ]
    for capability in sorted(set(fix_caps) | set(break_caps)):
        lines.append(f"| {capability} | {fix_caps[capability]} | {break_caps[capability]} |")
    lines += [
        "", "## 结论", "",
        f"1. Fix 主要集中于 `{fix_caps.most_common(1)[0][0] if fix_caps else '无'}`；Break 主要集中于 `{break_caps.most_common(1)[0][0] if break_caps else '无'}`。",
        "2. r3 的 OOD normal 增益不能单独视为视觉泛化；控制输入下新增高重合属于更关键的 break。",
        "3. 若 break 集中于 spatial/object/camera 相关样本，应优先设计 grounding-aware 评测与 reward，而不是继续无目标微调。",
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"fix_cases": len(fixes), "break_cases": len(breaks), "overlap_cases": overlap, "fix_capabilities": dict(fix_caps), "break_capabilities": dict(break_caps)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
