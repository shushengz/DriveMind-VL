
"""Build Stage 4 SFT-v3-r3 case gallery when r3 results exist."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Any

SPATIAL_RE=re.compile(r"left|right|front|back|lane|traffic light|pedestrian|vehicle|behind|ahead", re.I)


def read_csv(path:Path) -> list[dict[str,str]]:
    if not path.exists(): return []
    with path.open("r",encoding="utf-8",newline="") as f: return list(csv.DictReader(f))


def read_jsonl_map(path:Path) -> dict[str,dict[str,Any]]:
    out={}
    if not path.exists(): return out
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row=json.loads(line); out[str(row.get("id"))]=row
    return out


def flt(row:dict[str,str]|None,key:str) -> float:
    if not row: return 0.0
    try: return float(row.get(key,0) or 0)
    except Exception: return 0.0


def row_map(rows:list[dict[str,str]]) -> dict[str,dict[str,str]]:
    return {str(r.get("id")):r for r in rows}


def choose(row, sft2, r2, sample):
    q=str(sample.get("question", "")) if sample else ""
    if sft2 and flt(sft2,"normal_f1") >= 0.7 and flt(row,"normal_f1") >= 0.7: return "sft_v2_and_r3_correct"
    if sft2 and flt(row,"normal_f1") > flt(sft2,"normal_f1") + 0.2: return "sft_v2_wrong_r3_correct"
    if r2 and flt(row,"normal_f1") > flt(r2,"normal_f1") + 0.2: return "r2_wrong_r3_correct"
    if flt(row,"normal_f1") >= 0.7 and flt(row,"wrong_image_f1") >= flt(row,"normal_f1") - 0.05: return "r3_normal_ok_wrong_ok"
    if row.get("failure_type") == "blank_hallucination": return "r3_blank_hallucination"
    if flt(row,"text_only_f1") >= flt(row,"normal_f1"): return "r3_text_prior_bias"
    if row.get("failure_type") == "over_refusal": return "r3_over_refusal"
    if SPATIAL_RE.search(q) and flt(row,"normal_f1") < 0.5: return "r3_spatial_failure"
    return row.get("failure_type") or "unresolved"


def comment(kind):
    mapping={
        "sft_v2_and_r3_correct":"???? SFT-v2 ? r3 ??? normal ???????? r3 ????? normal QA ???",
        "sft_v2_wrong_r3_correct":"???? SFT-v2 ????? r3 ???? r3 answer-only calibration ??????",
        "r2_wrong_r3_correct":"???? r2 ????? r3 ??????? reason ??? control ????????",
        "r3_normal_ok_wrong_ok":"r3 ? normal ?????? wrong-image ??????????????????????",
        "r3_blank_hallucination":"r3 ? blank-image ???????????????? control calibration?",
        "r3_text_prior_bias":"r3 ? text-only ???????? normal??????????",
        "r3_over_refusal":"r3 ? normal ???????????????",
        "r3_spatial_failure":"??????????r3 normal ????????? grounding ?????",
    }
    return mapping.get(kind,"??????? r3 ? strict visual-control ? answer-only ???????")


def write_csv(path:Path, rows:list[dict[str,Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: path.write_text("",encoding="utf-8"); return
    fields=list(rows[0].keys())
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def write_html(path:Path, rows:list[dict[str,Any]]):
    fields=list(rows[0].keys()) if rows else ["message"]
    parts=["<html><head><meta charset='utf-8'><style>table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;max-width:360px}</style></head><body><h1>Stage 4 SFT-v3-r3 Case Gallery</h1><table><tr>"]
    parts += [f"<th>{html.escape(f)}</th>" for f in fields]+["</tr>"]
    for r in rows:
        parts.append("<tr>"); parts += [f"<td>{html.escape(str(r.get(f,'')))}</td>" for f in fields]; parts.append("</tr>")
    parts.append("</table></body></html>")
    path.write_text("".join(parts),encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--r3_cases", default="outputs/cases/lingoqa_sft_v3_r3_lingo_smoke_strict_visual_case_scores.csv")
    p.add_argument("--sft2_cases", default="outputs/cases/lingoqa_sft_v2_strict_visual_case_scores.csv")
    p.add_argument("--r2_cases", default="outputs/cases/lingoqa_sft_v3_r2_lingo_smoke_strict_visual_case_scores.csv")
    p.add_argument("--visual_samples", default="data/processed/visual_control/lingoqa_strict_normal.jsonl")
    p.add_argument("--output_csv", default="outputs/final_report/stage4_sft_v3_r3_case_gallery.csv")
    p.add_argument("--output_html", default="outputs/final_report/stage4_sft_v3_r3_case_gallery.html")
    p.add_argument("--limit", type=int, default=80)
    return p.parse_args()


def main():
    args=parse_args(); r3=read_csv(Path(args.r3_cases))
    if not r3:
        rows=[{"message":"r3 GPU training/eval has not been run yet."}]
        write_csv(Path(args.output_csv), rows); write_html(Path(args.output_html), rows)
        print(json.dumps({"csv":args.output_csv,"html":args.output_html,"count":0,"message":rows[0]["message"]},ensure_ascii=False,indent=2)); return
    sft2=row_map(read_csv(Path(args.sft2_cases))); r2=row_map(read_csv(Path(args.r2_cases))); samples=read_jsonl_map(Path(args.visual_samples))
    out=[]
    for row in sorted(r3, key=lambda r: flt(r,"case_gap")):
        sid=str(row.get("id")); sample=samples.get(sid,{})
        kind=choose(row,sft2.get(sid),r2.get(sid),sample)
        out.append({"id":sid,"case_type":kind,"question":sample.get("question",""),"gold":row.get("gold",""),"normal_f1":row.get("normal_f1",""),"text_only_f1":row.get("text_only_f1",""),"wrong_image_f1":row.get("wrong_image_f1",""),"blank_image_f1":row.get("blank_image_f1",""),"case_gap":row.get("case_gap",""),"failure_type":row.get("failure_type",""),"normal_prediction":row.get("normal_prediction",""),"chinese_comment":comment(kind)})
        if len(out)>=args.limit: break
    write_csv(Path(args.output_csv), out); write_html(Path(args.output_html), out)
    print(json.dumps({"csv":args.output_csv,"html":args.output_html,"count":len(out)},ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
