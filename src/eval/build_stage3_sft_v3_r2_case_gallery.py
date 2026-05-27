
"""Build a comparison-oriented Stage 3 case gallery."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

SPATIAL_RE = re.compile(r"left|right|front|back|lane|traffic light|pedestrian|vehicle|behind|ahead", re.I)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl_map(path: Path) -> dict[str, dict[str, Any]]:
    out={}
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line); out[str(row.get("id"))]=row
    return out


def flt(row: dict[str, str] | None, key: str) -> float:
    if not row: return 0.0
    try: return float(row.get(key, 0) or 0)
    except Exception: return 0.0


def row_map(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(r.get("id")): r for r in rows}


def comment(case_type: str) -> str:
    comments={
        "sft_v3_smoke_wrong_r2_better": "该样本中 SFT-v3-r2 相比 Stage 2 SFT-v3 smoke 明显改善，说明 normal anchor 和比例修复可能恢复了部分正常视觉问答能力。",
        "base_wrong_r2_better": "该样本中 Base 表现较差而 SFT-v3-r2 更好，说明 r2 对该类视觉问答有正向收益。",
        "r2_normal_ok_wrong_ok": "该样本中 SFT-v3-r2 normal 条件表现较好，但 wrong-image 条件也接近正确，仍可能存在语言先验或错误图像混淆。",
        "r2_blank_hallucination": "该样本中 blank-image 条件下仍出现视觉断言，说明控制条件下的幻觉没有完全消除。",
        "r2_text_prior_bias": "该样本中 text-only 条件接近或超过 normal，说明模型可能依赖语言先验。",
        "r2_over_refusal": "该样本中 normal 图像下发生拒答，需要检查是否出现过度校准。",
        "r2_spatial_failure": "该样本涉及空间关系，SFT-v3-r2 normal 表现不佳，说明空间 grounding 仍需加强。",
        "dpo_v7_better_than_r2": "该样本中 DPO-v7 明显优于 SFT-v3-r2，说明 r2 尚未完全恢复旧模型的 normal answer ability。",
    }
    return comments.get(case_type, "该样本用于观察 SFT-v3-r2 在 strict visual-control 协议下的表现。")


def choose_type(r2, base, sft3, dpo, sample) -> str:
    q=str(sample.get("question", "")) if sample else ""
    if sft3 and flt(r2,"normal_f1") > flt(sft3,"normal_f1") + 0.2:
        return "sft_v3_smoke_wrong_r2_better"
    if base and flt(r2,"normal_f1") > flt(base,"normal_f1") + 0.2:
        return "base_wrong_r2_better"
    if flt(r2,"normal_f1") >= 0.7 and flt(r2,"wrong_image_f1") >= flt(r2,"normal_f1") - 0.05:
        return "r2_normal_ok_wrong_ok"
    if r2.get("failure_type") == "blank_hallucination":
        return "r2_blank_hallucination"
    if flt(r2,"text_only_f1") >= flt(r2,"normal_f1"):
        return "r2_text_prior_bias"
    if r2.get("failure_type") == "over_refusal":
        return "r2_over_refusal"
    if SPATIAL_RE.search(q) and flt(r2,"normal_f1") < 0.5:
        return "r2_spatial_failure"
    if dpo and flt(dpo,"normal_f1") > flt(r2,"normal_f1") + 0.2:
        return "dpo_v7_better_than_r2"
    return str(r2.get("failure_type") or "unresolved")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields=list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    headers=list(rows[0].keys()) if rows else []
    parts=["<html><head><meta charset='utf-8'><style>body{font-family:Arial,sans-serif}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;max-width:360px}img{max-width:180px;max-height:120px}</style></head><body>", "<h1>Stage 3 SFT-v3-r2 Case Gallery</h1><table><thead><tr>"]
    parts += [f"<th>{html.escape(h)}</th>" for h in headers] + ["</tr></thead><tbody>"]
    for row in rows:
        parts.append("<tr>")
        for h in headers:
            val=row.get(h, "")
            if h == "image_paths" and val:
                imgs=[]
                for p in str(val).split(";")[:3]:
                    imgs.append(f"<div>{html.escape(p)}</div><img src='{html.escape(p)}'>")
                parts.append("<td>"+"".join(imgs)+"</td>")
            else:
                parts.append(f"<td>{html.escape(str(val))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--r2_cases", default="outputs/cases/lingoqa_sft_v3_r2_lingo_smoke_strict_visual_case_scores.csv")
    p.add_argument("--base_cases", default="outputs/cases/lingoqa_base_qwen25vl_3b_strict_visual_case_scores.csv")
    p.add_argument("--sft3_cases", default="outputs/cases/lingoqa_sft_v3_lingo_smoke_strict_visual_case_scores.csv")
    p.add_argument("--dpo_cases", default="outputs/cases/lingoqa_dpo_v7_strict_visual_case_scores.csv")
    p.add_argument("--visual_samples", default="data/processed/visual_control/lingoqa_strict_normal.jsonl")
    p.add_argument("--output_csv", default="outputs/final_report/stage3_sft_v3_r2_case_gallery.csv")
    p.add_argument("--output_html", default="outputs/final_report/stage3_sft_v3_r2_case_gallery.html")
    p.add_argument("--limit", type=int, default=80)
    return p.parse_args()


def main():
    args=parse_args()
    r2=read_csv(Path(args.r2_cases)); base=row_map(read_csv(Path(args.base_cases))); sft3=row_map(read_csv(Path(args.sft3_cases))); dpo=row_map(read_csv(Path(args.dpo_cases))); samples=read_jsonl_map(Path(args.visual_samples))
    gallery=[]
    for row in sorted(r2, key=lambda r: flt(r,"case_gap")):
        sid=str(row.get("id")); sample=samples.get(sid, {})
        ctype=choose_type(row, base.get(sid), sft3.get(sid), dpo.get(sid), sample)
        gallery.append({
            "id": sid,
            "case_type": ctype,
            "dataset": "lingoqa",
            "image_paths": ";".join(sample.get("image_paths") or []),
            "question": sample.get("question", ""),
            "gold": row.get("gold", ""),
            "r2_normal_f1": row.get("normal_f1", ""),
            "r2_text_only_f1": row.get("text_only_f1", ""),
            "r2_wrong_image_f1": row.get("wrong_image_f1", ""),
            "r2_blank_image_f1": row.get("blank_image_f1", ""),
            "r2_case_gap": row.get("case_gap", ""),
            "base_normal_f1": base.get(sid, {}).get("normal_f1", ""),
            "sft3_normal_f1": sft3.get(sid, {}).get("normal_f1", ""),
            "dpo_v7_normal_f1": dpo.get(sid, {}).get("normal_f1", ""),
            "r2_normal_prediction": row.get("normal_prediction", ""),
            "r2_text_only_prediction": row.get("text_only_prediction", ""),
            "r2_wrong_image_prediction": row.get("wrong_image_prediction", ""),
            "r2_blank_image_prediction": row.get("blank_image_prediction", ""),
            "chinese_comment": comment(ctype),
        })
        if len(gallery) >= args.limit:
            break
    write_csv(Path(args.output_csv), gallery); write_html(Path(args.output_html), gallery)
    print(json.dumps({"csv": args.output_csv, "html": args.output_html, "count": len(gallery)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
