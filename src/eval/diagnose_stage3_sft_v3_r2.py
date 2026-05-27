
"""Diagnose Stage 3 SFT-v3-r2 strict visual-control results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BASE = "base_qwen25vl_3b"
SFT2 = "sft_v2"
DPO7 = "dpo_v7"
SFT3 = "sft_v3_lingo_smoke"
R2 = "sft_v3_r2_lingo_smoke"


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def f(row: dict[str, Any] | None, key: str) -> float:
    if not row:
        return 0.0
    try:
        return float(row.get(key, 0) or 0)
    except Exception:
        return 0.0


def n(row: dict[str, Any] | None) -> int:
    if not row:
        return 0
    try:
        return int(float(row.get("num_samples", 0) or 0))
    except Exception:
        return 0


def find(rows: list[dict[str, Any]], model: str, dataset: str = "lingoqa") -> dict[str, Any] | None:
    return next((r for r in rows if r.get("dataset") == dataset and r.get("model_name") == model), None)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="outputs/final_report/stage3_sft_v3_r2_main_results.csv")
    p.add_argument("--output_json", default="outputs/final_report/stage3_sft_v3_r2_diagnosis.json")
    p.add_argument("--output_md", default="outputs/final_report/stage3_sft_v3_r2_diagnosis.md")
    args=p.parse_args()
    rows=read_rows(Path(args.input))
    base=find(rows, BASE); sft2=find(rows, SFT2); dpo7=find(rows, DPO7); sft3=find(rows, SFT3); r2=find(rows, R2)
    def flag(name: str, value: bool) -> dict[str, Any]: return {"name": name, "pass": bool(value)}
    checks=[]
    checks.append(flag("normal QA ability recovered vs Base", bool(r2 and base and f(r2,"normal_f1") >= f(base,"normal_f1"))))
    checks.append(flag("r2 better than Stage 2 SFT-v3 smoke", bool(r2 and sft3 and f(r2,"normal_f1") > f(sft3,"normal_f1"))))
    checks.append(flag("r2 close to SFT-v2", bool(r2 and sft2 and f(r2,"normal_f1") >= f(sft2,"normal_f1") - 0.02)))
    checks.append(flag("r2 reaches DPO-v7 band", bool(r2 and dpo7 and f(r2,"normal_f1") >= f(dpo7,"normal_f1") - 0.02)))
    checks.append(flag("visual dependency improved vs Base", bool(r2 and base and f(r2,"case_gap") >= f(base,"case_gap"))))
    checks.append(flag("control hallucination reduced vs Base", bool(r2 and base and f(r2,"control_hallucination_rate") < f(base,"control_hallucination_rate"))))
    checks.append(flag("no over-refusal vs Base", bool(r2 and base and f(r2,"normal_refusal_rate") <= f(base,"normal_refusal_rate") + 0.05)))
    language_prior = bool(r2 and base and all(f(r2,k)>f(base,k)+0.03 for k in ["text_only_f1","wrong_image_f1","blank_image_f1"]) and f(r2,"case_gap") <= f(base,"case_gap")+0.01)
    enter_dpo = bool(r2 and base and f(r2,"normal_f1") >= f(base,"normal_f1") and f(r2,"case_gap") >= f(base,"case_gap") - 0.02 and f(r2,"normal_refusal_rate") <= f(base,"normal_refusal_rate") + 0.05)
    enter_grpo = False
    report={"rows": rows, "checks": checks, "language_prior_strengthened": language_prior, "recommend_enter_dpo_v8": enter_dpo, "recommend_enter_grpo_lite": enter_grpo}
    yes = "\u662f"
    no = "\u5426"
    lines=["# Stage 3 SFT-v3-r2 \u8bca\u65ad", ""]
    if not r2:
        lines.append("\u672a\u627e\u5230 SFT-v3-r2 \u7ed3\u679c\uff0c\u65e0\u6cd5\u8bca\u65ad\u3002")
    else:
        lines.append(f"SFT-v3-r2 Normal F1={f(r2,'normal_f1'):.4f}, Case Gap={f(r2,'case_gap'):.4f}, hallucination={f(r2,'control_hallucination_rate'):.4f}, normal refusal={f(r2,'normal_refusal_rate'):.4f}, \u6837\u672c\u6570={n(r2)}\u3002")
        if base: lines.append(f"Base Normal F1={f(base,'normal_f1'):.4f}, Case Gap={f(base,'case_gap'):.4f}, hallucination={f(base,'control_hallucination_rate'):.4f}\u3002")
        if sft3: lines.append(f"Stage 2 SFT-v3 smoke Normal F1={f(sft3,'normal_f1'):.4f}, Case Gap={f(sft3,'case_gap'):.4f}\u3002")
        if sft2: lines.append(f"SFT-v2 Normal F1={f(sft2,'normal_f1'):.4f}\u3002")
        if dpo7: lines.append(f"DPO-v7 Normal F1={f(dpo7,'normal_f1'):.4f}\u3002")
    lines += ["", "## \u5224\u65ad"]
    for item in checks:
        lines.append(f"- {item['name']}: {yes if item['pass'] else no}")
    if language_prior:
        lines.append("- \u98ce\u9669\uff1atext/wrong/blank control F1 \u540c\u65f6\u4e0a\u5347\u4f46 case gap \u672a\u63d0\u5347\uff0c\u53ef\u80fd\u53ea\u662f\u8bed\u8a00\u5148\u9a8c\u589e\u5f3a\u3002")
    lines += ["", "## \u51b3\u7b56"]
    lines.append(f"\u662f\u5426\u5efa\u8bae\u8fdb\u5165 DPO-v8\uff1a{yes if enter_dpo else no}\u3002")
    lines.append("\u662f\u5426\u5efa\u8bae\u8fdb\u5165 GRPO-lite\uff1a\u5426\u3002\u53ea\u6709 DPO-v8 \u540e\u4ecd\u6709\u660e\u663e control hallucination\uff0c\u624d\u8003\u8651 GRPO-lite\u3002")
    if enter_dpo:
        lines.append("\u4e0b\u4e00\u6b65\u5efa\u8bae\uff1a\u7528 r2 \u4f5c\u4e3a\u66f4\u7a33\u7684 SFT init\uff0c\u6784\u9020 DPO-v8\uff0c\u5e76\u4fdd\u7559 strict eval \u4f5c\u4e3a\u56de\u5f52\u95e8\u69db\u3002")
    else:
        lines.append("\u4e0b\u4e00\u6b65\u5efa\u8bae\uff1a\u7ee7\u7eed\u8c03\u6574 r2 \u6570\u636e\u6bd4\u4f8b\u6216\u8bad\u7ec3\u6b65\u6570\uff0c\u4f18\u5148\u4fdd\u8bc1 Normal F1 \u4e0d\u4f4e\u4e8e Base\u3002")
    out_json=Path(args.output_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"json": args.output_json, "md": args.output_md, "enter_dpo_v8": enter_dpo}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
