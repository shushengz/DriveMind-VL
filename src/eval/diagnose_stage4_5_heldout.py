"""Diagnose Stage 4.5 held-out results after leakage audit."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

BASE = "base_qwen25vl_3b"
SFT2 = "sft_v2"
R2 = "sft_v3_r2_lingo_smoke"
R3 = "sft_v3_r3_lingo_smoke"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def answer_row(rows: list[dict[str, str]], model: str) -> dict[str, str] | None:
    return next((row for row in rows if row.get("model_name") == model and row.get("score_mode") == "answer_only"), None)


def number(row: dict[str, str] | None, key: str) -> float:
    return float(row.get(key, 0) or 0) if row else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4.5 held-out diagnostic report.")
    parser.add_argument("--overlap", default="outputs/final_report/stage4_5_train_eval_overlap.json")
    parser.add_argument("--results", default="outputs/final_report/stage4_5_heldout_main_results.csv")
    parser.add_argument("--output_json", default="outputs/final_report/stage4_5_heldout_diagnosis.json")
    parser.add_argument("--output_md", default="outputs/final_report/stage4_5_heldout_diagnosis.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overlap_path = Path(args.overlap)
    overlap = json.loads(overlap_path.read_text(encoding="utf-8")) if overlap_path.exists() else {}
    rows = read_csv(Path(args.results))
    base, sft2, r2, r3 = (answer_row(rows, model) for model in (BASE, SFT2, R2, R3))
    heldout_ready = all((base, sft2, r2, r3))
    leakage = bool(overlap.get("overlap_exists"))
    yes = "\u662f"
    no = "\u5426"
    out: dict[str, Any] = {
        "overlap": overlap,
        "heldout_results_available": heldout_ready,
        "stage4_may_be_overestimated": leakage,
        "recommend_enter_dpo_v8": False,
        "recommend_enter_grpo_lite": False,
    }
    lines = ["# Stage 4.5 Held-out \u8bca\u65ad", ""]
    lines += [
        f"- Stage 4 \u662f\u5426\u5b58\u5728 train/eval overlap\uff1a{yes if leakage else no}",
        f"- overlap count\uff1a{overlap.get('overlap_count', 'N/A')}",
        f"- overlap rate (eval)\uff1a{float(overlap.get('overlap_rate_eval', 0)):.2%}",
        f"- Stage 4 \u539f\u59cb r3 \u7ed3\u679c\u662f\u5426\u53ef\u80fd\u88ab\u9ad8\u4f30\uff1a{yes if leakage else no}",
    ]
    if not heldout_ready:
        out["message"] = "Held-out eval results are not available; DPO-v8 decision is blocked."
        lines += [
            "",
            "## \u5f53\u524d\u963b\u585e",
            "",
            "\u5c1a\u65e0\u5b8c\u6574 held-out \u8bc4\u6d4b\u7ed3\u679c\uff0c\u4e0d\u80fd\u5224\u5b9a r3 \u662f\u5426\u53ef\u4f5c\u4e3a DPO-v8 \u521d\u59cb checkpoint\u3002",
            "\u5fc5\u987b\u5148\u6784\u5efa\u4e0e r3 \u8bad\u7ec3 ID \u65e0\u91cd\u53e0\u7684 strict visual-control \u6837\u672c\u6c60\uff0c\u7136\u540e\u91cd\u8dd1 Base / SFT-v2 / DPO-v7 / r2 / r3 \u5bf9\u6bd4\u3002",
            "",
            "## \u51b3\u7b56",
            "",
            "- \u662f\u5426\u5efa\u8bae\u8fdb\u5165 DPO-v8\uff1a\u5426\uff0c\u5f53\u524d\u88ab held-out \u8bc4\u6d4b\u7f3a\u5931\u963b\u585e\u3002",
            "- \u662f\u5426\u5efa\u8bae\u8fdb\u5165 GRPO-lite\uff1a\u5426\u3002",
        ]
    else:
        checks = {
            "r3_normal_ge_base": number(r3, "normal_f1") >= number(base, "normal_f1"),
            "r3_normal_close_or_ge_sft2": number(r3, "normal_f1") >= number(sft2, "normal_f1") - 0.03,
            "r3_case_gap_ge_base": number(r3, "case_gap") >= number(base, "case_gap"),
            "r3_hallucination_le_sft2_or_r2": number(r3, "control_hallucination_rate") <= number(sft2, "control_hallucination_rate") or number(r3, "control_hallucination_rate") <= number(r2, "control_hallucination_rate"),
            "r3_no_over_refusal": number(r3, "normal_refusal_rate") <= number(base, "normal_refusal_rate") + 0.05,
        }
        enter = all(checks.values())
        out.update({"checks": checks, "recommend_enter_dpo_v8": enter})
        lines += [
            "",
            "## Answer-only Held-out \u7ed3\u679c",
            "",
            f"- Base Normal F1: {number(base, 'normal_f1'):.4f}",
            f"- SFT-v2 Normal F1: {number(sft2, 'normal_f1'):.4f}",
            f"- SFT-v3-r2 Normal F1: {number(r2, 'normal_f1'):.4f}",
            f"- SFT-v3-r3 Normal F1: {number(r3, 'normal_f1'):.4f}",
            f"- r3 Case Gap: {number(r3, 'case_gap'):.4f}",
            f"- r3 Control Hallucination: {number(r3, 'control_hallucination_rate'):.4f}",
            f"- r3 Normal Refusal: {number(r3, 'normal_refusal_rate'):.4f}",
            "",
            "## \u5224\u65ad",
            "",
            f"- held-out \u4e0a r3 \u662f\u5426\u4ecd\u4f18\u4e8e Base\uff1a{yes if checks['r3_normal_ge_base'] else no}\u3002",
            f"- held-out \u4e0a r3 \u662f\u5426\u63a5\u8fd1\u6216\u8d85\u8fc7 SFT-v2\uff1a{yes if checks['r3_normal_close_or_ge_sft2'] else no}\u3002",
            f"- held-out \u4e0a r3 Case Gap \u662f\u5426\u4f18\u4e8e Base\uff1a{yes if checks['r3_case_gap_ge_base'] else no}\u3002r3={number(r3, 'case_gap'):.4f}, Base={number(base, 'case_gap'):.4f}\u3002",
            f"- held-out \u4e0a r3 hallucination \u662f\u5426\u4f4e\u4e8e\u6216\u7b49\u4e8e SFT-v2/r2\uff1a{yes if checks['r3_hallucination_le_sft2_or_r2'] else no}\u3002r3={number(r3, 'control_hallucination_rate'):.4f}, SFT-v2={number(sft2, 'control_hallucination_rate'):.4f}, r2={number(r2, 'control_hallucination_rate'):.4f}\u3002",
            f"- held-out \u4e0a r3 normal refusal \u662f\u5426\u5f02\u5e38\uff1a{no if checks['r3_no_over_refusal'] else yes}\u3002",
            "## \u51b3\u7b56",
            "",
            f"- \u662f\u5426\u5efa\u8bae\u8fdb\u5165 DPO-v8\uff1a{yes if enter else no}\u3002",
            "- \u662f\u5426\u5efa\u8bae\u8fdb\u5165 GRPO-lite\uff1a\u5426\uff0c\u5fc5\u987b\u5148\u5b8c\u6210 DPO-v8 \u9a8c\u8bc1\u3002",
            "",
            "## \u4e0b\u4e00\u6b65\u5efa\u8bae",
            "",
            "\u4fdd\u7559 r3 \u4f5c\u4e3a normal QA \u8f83\u5f3a\u7684\u5019\u9009 checkpoint\uff0c\u4f46\u5148\u68c0\u67e5 held-out \u4e0a blank/text-only/wrong-image \u5e7b\u89c9\u6848\u4f8b\uff0c\u8c03\u6574\u4f4e\u98ce\u9669 control calibration \u6216 preference \u8bbe\u8ba1\uff0c\u7136\u540e\u518d\u91cd\u65b0\u8bc4\u4f30 DPO-v8 \u5165\u53e3\u6761\u4ef6\u3002",
        ]
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"heldout_results_available": heldout_ready, "recommend_enter_dpo_v8": out["recommend_enter_dpo_v8"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
