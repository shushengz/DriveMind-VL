"""Build final offline ablation tables from consolidated result files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

COLUMNS = ["group", "ablation", "purpose", "comparison", "key_results", "conclusion", "in_final_method"]


def rows_by_model(path: Path) -> dict[str, dict[str, str]]:
    return {row["model_name"]: row for row in csv.DictReader(path.open(encoding="utf-8", newline=""))}


def n(row: dict[str, str], key: str) -> float:
    return float(row.get(key, 0) or 0)


def fmt(row: dict[str, str], fields: list[str]) -> str:
    return ", ".join(f"{key}={n(row, key):.4f}" for key in fields)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final DriveMind-VL ablation tables.")
    parser.add_argument("--answer_only", default="outputs/final_report/final_main_results_answer_only.csv")
    parser.add_argument("--raw", default="outputs/final_report/final_main_results_raw.csv")
    parser.add_argument("--pre_heldout", default="outputs/final_report/stage4_sft_v3_r3_main_results.csv")
    parser.add_argument("--output_csv", default="outputs/final_report/final_ablation_results.csv")
    parser.add_argument("--output_md", default="outputs/final_report/final_ablation_results.md")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    answer = rows_by_model(Path(args.answer_only))
    raw = rows_by_model(Path(args.raw))
    r3 = answer["SFT-v3-r3"]
    r2 = answer["SFT-v3-r2"]
    dpo8, dpo81, dpo82 = answer["DPO-v8 rule-based"], answer["DPO-v8.1 step-50"], answer["DPO-v8.2 step-25"]
    pre_note = "Stage 4 pre-heldout table is excluded from final claims."
    if Path(args.pre_heldout).exists():
        pre_note += " It remains an audit trail only."
    rows: list[dict[str, Any]] = [
        {
            "group": "Evaluation Protocol", "ablation": "raw_full vs answer_only",
            "purpose": "Remove JSON/reason formatting effects from QA scoring.",
            "comparison": "SFT-v3-r3 raw_full vs answer_only",
            "key_results": f"raw normal_f1={n(raw['SFT-v3-r3'],'normal_f1'):.4f}; answer-only normal_f1={n(r3,'normal_f1'):.4f}",
            "conclusion": "Answer-only scoring is necessary for interpretable final comparison.",
            "in_final_method": "Yes",
        },
        {
            "group": "Evaluation Protocol", "ablation": "leaked/pre-heldout vs held-out",
            "purpose": "Prevent optimistic model selection caused by train/eval overlap.",
            "comparison": "Stage 4 pre-heldout outputs vs Stage 4.5+ held-out 100 IDs",
            "key_results": pre_note,
            "conclusion": "Only held-out strict_visual predictions are eligible for final claims.",
            "in_final_method": "Yes",
        },
        {
            "group": "Evaluation Protocol", "ablation": "normal F1 vs visual-control metrics",
            "purpose": "Test whether good normal answers depend on visual evidence.",
            "comparison": "SFT-v3-r3 normal_f1 together with case/control metrics",
            "key_results": fmt(r3, ["normal_f1", "case_gap", "control_high_f1_rate_0_20", "control_hallucination_rate"]),
            "conclusion": "Normal F1 alone is insufficient; controls expose remaining prior-answer behavior.",
            "in_final_method": "Yes",
        },
        {
            "group": "SFT", "ablation": "SFT-v3-r2 vs SFT-v3-r3",
            "purpose": "Balance normal replay and control calibration.",
            "comparison": "r2 control-heavy recipe vs r3 answer-only calibration recipe",
            "key_results": f"r2: {fmt(r2,['normal_f1','case_gap'])}; r3: {fmt(r3,['normal_f1','case_gap'])}",
            "conclusion": "r3 is the stable SFT branch selected as the final checkpoint.",
            "in_final_method": "SFT-v3-r3 only",
        },
        {
            "group": "SFT", "ablation": "higher normal replay in r3",
            "purpose": "Protect useful visual QA behavior while applying control calibration.",
            "comparison": "SFT-v3-r2 vs SFT-v3-r3 normal performance",
            "key_results": f"normal_f1 rises from {n(r2,'normal_f1'):.4f} to {n(r3,'normal_f1'):.4f}; normal_refusal remains {n(r3,'normal_refusal_rate'):.4f}",
            "conclusion": "Normal anchors are necessary to avoid calibration-induced capability loss.",
            "in_final_method": "Yes",
        },
        {
            "group": "SFT", "ablation": "lower control calibration intensity in r3",
            "purpose": "Avoid turning visual-dependency calibration into blanket conservatism.",
            "comparison": "SFT-v3-r2 vs SFT-v3-r3 case behavior",
            "key_results": f"case_gap improves from {n(r2,'case_gap'):.4f} to {n(r3,'case_gap'):.4f}",
            "conclusion": "A moderated control calibration recipe yields the most stable held-out balance.",
            "in_final_method": "Yes",
        },
        {
            "group": "SFT", "ablation": "answer-only output calibration",
            "purpose": "Reduce format/reason interference while preserving normal answers.",
            "comparison": "SFT-v3-r3 evaluated under answer-only protocol",
            "key_results": fmt(r3, ["normal_f1", "case_gap", "normal_refusal_rate"]),
            "conclusion": "Answer-only calibration plus held-out scoring produces the strongest reliable candidate.",
            "in_final_method": "Yes",
        },
        {
            "group": "Preference", "ablation": "DPO-v8 rule-based preference",
            "purpose": "Reduce explicit control hallucination using constructed cautious preferences.",
            "comparison": "DPO-v8 vs SFT-v3-r3",
            "key_results": f"r3 hallucination={n(r3,'control_hallucination_rate'):.4f}, case_gap={n(r3,'case_gap'):.4f}; v8 hallucination={n(dpo8,'control_hallucination_rate'):.4f}, case_gap={n(dpo8,'case_gap'):.4f}",
            "conclusion": "Hallucination drops slightly, but case gap regresses.",
            "in_final_method": "No; ablation only",
        },
        {
            "group": "Preference", "ablation": "DPO-v8.1 model-mined preference",
            "purpose": "Use actual model failures as rejected responses.",
            "comparison": "DPO-v8.1 step-50 vs SFT-v3-r3",
            "key_results": fmt(dpo81, ["normal_f1", "case_gap", "blank_high_f1_rate_0_20", "control_hallucination_rate"]),
            "conclusion": "Explicit hallucination improves slightly while blank-image prior and case gap worsen.",
            "in_final_method": "No; failure analysis",
        },
        {
            "group": "Preference", "ablation": "DPO-v8.2 case-gap-aware preference",
            "purpose": "Target blank/control high-F1 overlap identified in Stage 8.5.",
            "comparison": "DPO-v8.2 step-25 vs SFT-v3-r3",
            "key_results": f"r3: {fmt(r3,['case_gap','blank_high_f1_rate_0_20','control_high_f1_rate_0_20'])}; v8.2: {fmt(dpo82,['case_gap','blank_high_f1_rate_0_20','control_high_f1_rate_0_20'])}",
            "conclusion": "The targeted preference does not resolve high-F1 controls and is not selected.",
            "in_final_method": "No; ablation only",
        },
        {
            "group": "Failure Mode", "ablation": "hallucination rate vs case_gap",
            "purpose": "Check whether lower explicit hallucination implies stronger visual dependency.",
            "comparison": "r3 vs DPO-v8/v8.1/v8.2",
            "key_results": "All DPO variants can lower hallucination slightly while producing worse case_gap than r3.",
            "conclusion": "Hallucination is necessary but not sufficient as a calibration metric.",
            "in_final_method": "Diagnostic metric suite",
        },
        {
            "group": "Failure Mode", "ablation": "blank-image high-F1 prior answer",
            "purpose": "Measure answer overlap despite absent visual evidence.",
            "comparison": "r3 vs DPO-v8.1 step-50 vs DPO-v8.2",
            "key_results": f"r3={n(r3,'blank_high_f1_rate_0_20'):.4f}; v8.1-50={n(dpo81,'blank_high_f1_rate_0_20'):.4f}; v8.2={n(dpo82,'blank_high_f1_rate_0_20'):.4f}",
            "conclusion": "Blank-image prior answer remains the central unresolved failure mode.",
            "in_final_method": "Diagnostic only",
        },
        {
            "group": "Failure Mode", "ablation": "control direct answer / wrong-image confound",
            "purpose": "Reveal plausible answers emitted under corrupted evidence.",
            "comparison": "All held-out control settings",
            "key_results": f"r3 direct_answer={n(r3,'control_direct_answer_rate'):.4f}; v8.2 direct_answer={n(dpo82,'control_direct_answer_rate'):.4f}",
            "conclusion": "Future reward design must penalize confident control answers without causing normal refusal.",
            "in_final_method": "Future reward target",
        },
    ]
    if args.dry_run:
        print(json.dumps({"ablation_rows": len(rows)}, ensure_ascii=False, indent=2))
        return
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    md = ["# Final Ablation Results", "", "| Group | Ablation | Key Result | Conclusion | Final Method |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        md.append(f"| {row['group']} | {row['ablation']} | {row['key_results']} | {row['conclusion']} | {row['in_final_method']} |")
    Path(args.output_md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output_csv, "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
