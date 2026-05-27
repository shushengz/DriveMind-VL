"""Diagnose Stage 2 strict visual-control results."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any

BASE_MODEL_NAME = "base_qwen25vl_3b"

def f(row: dict[str, Any] | None, key: str) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0

def n(row: dict[str, Any] | None, key: str = "num_samples") -> int:
    if not row:
        return 0
    try:
        return int(float(row.get(key, 0) or 0))
    except Exception:
        return 0

def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))

def find_exact(rows: list[dict[str, Any]], dataset: str, model_name: str) -> dict[str, Any] | None:
    return next((r for r in rows if r.get("dataset") == dataset and r.get("model_name") == model_name), None)

def diagnose_pair(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, bool]:
    return {
        "normal_answer_ability_improved": f(cand, "normal_f1") > f(base, "normal_f1"),
        "visual_dependency_improved": f(cand, "case_gap") > f(base, "case_gap"),
        "language_prior_strengthened": all(f(cand, k) > f(base, k) + 0.03 for k in ["text_only_f1", "wrong_image_f1", "blank_image_f1"]) and f(cand, "case_gap") <= f(base, "case_gap") + 0.01,
        "over_refusal_risk": f(cand, "normal_refusal_rate") > f(base, "normal_refusal_rate") + 0.05,
        "hallucination_reduced": f(cand, "control_hallucination_rate") < f(base, "control_hallucination_rate") - 0.03,
        "possible_over_calibration": f(cand, "normal_f1") < f(base, "normal_f1") and f(cand, "case_gap") > f(base, "case_gap"),
        "possible_length_verbosity_issue": f(cand, "avg_answer_length") > f(base, "avg_answer_length") * 1.4 + 5,
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/final_report/stage2_strict_eval_main_results.csv")
    p.add_argument("--output_json", default="outputs/final_report/stage2_diagnosis.json")
    p.add_argument("--output_md", default="outputs/final_report/stage2_diagnosis.md")
    p.add_argument("--min_samples_for_decision", type=int, default=30)
    args = p.parse_args()
    rows = read_rows(Path(args.input))
    diagnosis: dict[str, Any] = {"rows": rows, "datasets": {}, "recommendations": {}, "min_samples_for_decision": args.min_samples_for_decision}
    lines = ["# Stage 2 Strict Visual-Control \u8bca\u65ad", ""]
    if any(n(r) < args.min_samples_for_decision for r in rows):
        lines += [f"\u6ce8\u610f\uff1a\u5b58\u5728\u5c11\u4e8e {args.min_samples_for_decision} \u4e2a case \u7684 micro/smoke \u7ed3\u679c\uff0c\u53ea\u7528\u4e8e\u94fe\u8def\u9a8c\u8bc1\uff0c\u4e0d\u4f5c\u4e3a\u8fdb\u5165\u4e0b\u4e00\u9636\u6bb5\u7684\u4f9d\u636e\u3002", ""]
    for dataset in sorted({r.get("dataset", "") for r in rows if r.get("dataset")}):
        ds_rows = [r for r in rows if r.get("dataset") == dataset]
        decision_rows = [r for r in ds_rows if n(r) >= args.min_samples_for_decision]
        base = find_exact(rows, dataset, BASE_MODEL_NAME)
        cands = [r for r in ds_rows if "sft_v3" in r.get("model_name", "")]
        best = max(decision_rows or ds_rows, key=lambda r: f(r, "normal_f1"), default=None)
        dres: dict[str, Any] = {"best_model_by_normal_f1": best.get("model_name") if best else "", "comparisons": {}}
        lines.append(f"## {dataset}")
        if best:
            lines.append(f"\u5f53\u524d Normal F1 \u6700\u9ad8\u6a21\u578b\uff1a{best.get('model_name')}\uff0cNormal F1={f(best,'normal_f1'):.4f}\uff0c\u6837\u672c\u6570={n(best)}\u3002")
        if not base or n(base) < args.min_samples_for_decision:
            lines.append("\u672a\u627e\u5230\u8db3\u91cf Base \u7ed3\u679c\uff0c\u65e0\u6cd5\u5224\u65ad SFT-v3 \u662f\u5426\u4f18\u4e8e Base\u3002")
        for cand in cands:
            comp = diagnose_pair(base, cand) if base and n(base) >= args.min_samples_for_decision else {}
            dres["comparisons"][cand.get("model_name", "")] = comp
            lines.append(f"- {cand.get('model_name')}: normal_f1={f(cand,'normal_f1'):.4f}, case_gap={f(cand,'case_gap'):.4f}, normal_refusal_rate={f(cand,'normal_refusal_rate'):.4f}, \u6837\u672c\u6570={n(cand)}\u3002")
            if n(cand) < args.min_samples_for_decision:
                lines.append("  - \u6837\u672c\u6570\u4e0d\u8db3\uff0c\u4ec5\u8bf4\u660e\u6d41\u7a0b\u8dd1\u901a\uff0c\u4e0d\u5224\u65ad\u4f18\u52a3\u3002")
            elif comp:
                if comp.get("normal_answer_ability_improved"):
                    lines.append("  - Normal answer ability improved\u3002")
                else:
                    lines.append("  - Normal answer ability \u672a\u8d85\u8fc7 Base\u3002")
                if comp.get("visual_dependency_improved"):
                    lines.append("  - visual grounding / case gap improved\u3002")
                if comp.get("hallucination_reduced"):
                    lines.append("  - control hallucination reduced\u3002")
                if comp.get("language_prior_strengthened"):
                    lines.append("  - \u98ce\u9669\uff1acontrol F1 \u540c\u65f6\u4e0a\u5347\u4f46 case gap \u6ca1\u63d0\u5347\uff0c\u53ef\u80fd\u53ea\u662f\u8bed\u8a00\u5148\u9a8c\u589e\u5f3a\u3002")
                if comp.get("over_refusal_risk"):
                    lines.append("  - \u98ce\u9669\uff1anormal \u4e0b\u62d2\u7b54\u7387\u660e\u663e\u4e0a\u5347\uff0c\u5b58\u5728\u8fc7\u5ea6\u62d2\u7b54\u3002")
                if comp.get("possible_over_calibration"):
                    lines.append("  - \u98ce\u9669\uff1aNormal F1 \u4e0b\u964d\u4f46 gap \u4e0a\u5347\uff0c\u53ef\u80fd\u8fc7\u5ea6\u6821\u51c6\u3002")
        if dataset == "drivelm" and base and not cands:
            dres["ood_generalization_issue"] = True
            lines.append("DriveLM \u76ee\u524d\u53ea\u6709 Base \u8db3\u91cf\u7ed3\u679c\uff0c\u6682\u4e0d\u5224\u65ad SFT-v3 OOD \u6cdb\u5316\u3002")
        diagnosis["datasets"][dataset] = dres
        lines.append("")
    sft_rows = [r for r in rows if "sft_v3" in r.get("model_name", "") and n(r) >= args.min_samples_for_decision]
    dpo_ok = bool(sft_rows)
    for row in sft_rows:
        base = find_exact(rows, row.get("dataset", ""), BASE_MODEL_NAME)
        if not base or n(base) < args.min_samples_for_decision:
            dpo_ok = False
            continue
        if f(row, "normal_f1") < f(base, "normal_f1") or f(row, "normal_refusal_rate") > f(base, "normal_refusal_rate") + 0.05 or f(row, "case_gap") < f(base, "case_gap") - 0.02:
            dpo_ok = False
    diagnosis["recommendations"]["enter_dpo_v8"] = dpo_ok
    diagnosis["recommendations"]["enter_grpo_lite"] = False
    dpo_text = "\u662f" if dpo_ok else "\u5426\u6216\u9700\u8981\u8865\u5145\u9a8c\u8bc1"
    lines += ["## \u51b3\u7b56\u5efa\u8bae", f"\u662f\u5426\u5efa\u8bae\u8fdb\u5165 DPO-v8\uff1a{dpo_text}\u3002", "\u662f\u5426\u5efa\u8bae\u8fdb\u5165 GRPO-lite\uff1a\u6682\u4e0d\u5efa\u8bae\u3002\u9700\u8981\u5148\u786e\u8ba4 reward harness \u80fd\u7a33\u5b9a\u533a\u5206 normal/control\uff0c\u4e14 SFT-v3 \u6216 DPO-v8 \u63d0\u4f9b\u7a33\u5b9a init checkpoint\u3002", "\u4e0b\u4e00\u6b65\u5efa\u8bae\uff1a\u4fee\u6b63 SFT-v3 \u6570\u636e\u6bd4\u4f8b\u5e76\u6269\u5927 strict eval \u6837\u672c\u6570\uff0c\u590d\u6838 raw predictions \u4e0e case gallery\uff0c\u518d\u51b3\u5b9a DPO-v8\u3002"]
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": args.output_json, "md": args.output_md}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
