
"""Audit SFT-v3-r3 answer-only data."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CONTROL_TYPES={"blank_calibration","wrong_image_calibration","text_only_calibration"}
NORMAL_TYPES={"normal_replay","spatial_normal_qa","extra_normal_fill"}
REFUSAL_RE=re.compile(r"\u65e0\u6cd5\u5224\u65ad|\u65e0\u6cd5\u786e\u5b9a|\u7f3a\u5c11\u56fe\u50cf\u4fe1\u606f|cannot determine|not enough information|unable to determine", re.I)
FORBIDDEN_RE=re.compile(r"\b(image|visual|visible|scene|frame|picture)\b|\u56fe\u50cf|\u753b\u9762|\u56fe\u4e2d|\u89c6\u89c9", re.I)
PATH_RE=re.compile(r"(?:^|[\s\"'])/?(?:root/|data/|outputs/|checkpoints/|src/)[^\s\"']+", re.I)
BAD_FIELDS={"reason","references","category","subcategory","debug","metadata"}


def read_jsonl(path:Path) -> list[dict[str,Any]]:
    rows=[]
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def assistant(row:dict[str,Any]) -> tuple[dict[str,Any]|None,bool]:
    obj=row.get("assistant")
    if isinstance(obj,dict): return obj, set(obj.keys()) == {"answer"}
    try:
        msg=(row.get("messages") or [])[-1]
        parsed=json.loads(str(msg.get("content", "")))
        return parsed, isinstance(parsed,dict) and set(parsed.keys()) == {"answer"}
    except Exception:
        return None, False


def length(text:str) -> int:
    return len(text.split()) if re.search(r"[A-Za-z]", text or "") else len(text or "")


def audit(rows:list[dict[str,Any]], seed:int=42) -> dict[str,Any]:
    total=len(rows); counts=Counter(str(r.get("sample_type") or "unknown") for r in rows)
    normal_like=sum(counts[t] for t in NORMAL_TYPES); control=sum(counts[t] for t in CONTROL_TYPES)
    valid=0; reason=0; normal_refusal=0; forbidden=0; leaks=0; lengths=[]; samples=defaultdict(list)
    shuffled=list(rows); random.Random(seed).shuffle(shuffled)
    for row in shuffled:
        st=str(row.get("sample_type") or "unknown")
        obj, ok=assistant(row)
        if len(samples[st]) < 5:
            samples[st].append({"id": row.get("id"), "assistant": obj, "prompt": str(row.get("prompt", ""))[:180]})
    for row in rows:
        st=str(row.get("sample_type") or "unknown")
        obj, ok=assistant(row)
        if ok: valid += 1
        if not isinstance(obj,dict):
            leaks += 1; continue
        if "reason" in obj: reason += 1
        if any(k in BAD_FIELDS for k in obj): leaks += 1
        ans=str(obj.get("answer", ""))
        lengths.append(length(ans))
        if st in NORMAL_TYPES and REFUSAL_RE.search(ans): normal_refusal += 1
        if st in CONTROL_TYPES and FORBIDDEN_RE.search(ans): forbidden += 1
        if PATH_RE.search(json.dumps(obj,ensure_ascii=False)): leaks += 1
    report={
        "total": total,
        "sample_type_counts": dict(counts),
        "sample_type_ratios": {k:v/total for k,v in counts.items()} if total else {},
        "normal_like_ratio": normal_like/total if total else 0.0,
        "control_ratio": control/total if total else 0.0,
        "output_json_valid_rate": valid/total if total else 0.0,
        "avg_answer_length": sum(lengths)/len(lengths) if lengths else 0.0,
        "normal_refusal_rate": normal_refusal/normal_like if normal_like else 0.0,
        "control_forbidden_visual_terms_rate": forbidden/control if control else 0.0,
        "reason_field_count": reason,
        "metadata_leak_count": leaks,
        "samples_by_type": dict(samples),
    }
    fail=[]
    if report["normal_like_ratio"] < 0.75: fail.append("normal_replay + spatial_normal_qa below 75%")
    if report["control_ratio"] > 0.15: fail.append("control ratio above 15%")
    if report["normal_refusal_rate"] >= 0.01: fail.append("normal refusal rate >= 1%")
    if report["output_json_valid_rate"] <= 0.98: fail.append("output JSON valid rate <= 98%")
    if report["reason_field_count"] != 0: fail.append("reason field appears")
    if report["control_forbidden_visual_terms_rate"] >= 0.02: fail.append("control forbidden visual terms >= 2%")
    if report["metadata_leak_count"] != 0: fail.append("metadata leaks in assistant output")
    report["train_ready"] = not fail
    report["fail_reasons"] = fail
    return report


def write_md(path:Path, report:dict[str,Any]) -> None:
    lines=["# SFT-v3-r3 Answer-only ????", "", f"?????{report['total']}", "", "## ????"]
    for key in ["normal_like_ratio","control_ratio","output_json_valid_rate","normal_refusal_rate","control_forbidden_visual_terms_rate"]:
        lines.append(f"- {key}: {report[key]:.4f}")
    lines.append(f"- reason_field_count: {report['reason_field_count']}")
    lines.append(f"- metadata_leak_count: {report['metadata_leak_count']}")
    lines += ["", "## ????"]
    for k,v in sorted(report["sample_type_counts"].items()):
        lines.append(f"- {k}: {v} ({report['sample_type_ratios'].get(k,0):.4f})")
    lines += ["", "## ????", "??" if report["train_ready"] else "???"]
    for reason in report["fail_reasons"]: lines.append(f"- {reason}")
    lines += ["", "## ??????"]
    for st,items in sorted(report["samples_by_type"].items()):
        lines.append(f"### {st}")
        for item in items: lines.append(f"- {item['id']}: {json.dumps(item['assistant'], ensure_ascii=False)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--input", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    p.add_argument("--output_json", default="outputs/data_audit/sft_v3_r3_lingo_audit.json")
    p.add_argument("--output_md", default="outputs/data_audit/sft_v3_r3_lingo_audit.md")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args=parse_args(); report=audit(read_jsonl(Path(args.input)), args.seed)
    out=Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    write_md(Path(args.output_md), report)
    print(json.dumps({"json":args.output_json,"md":args.output_md,"train_ready":report["train_ready"],"fail_reasons":report["fail_reasons"]},ensure_ascii=False,indent=2))
    if not report["train_ready"]: raise SystemExit(1)

if __name__ == "__main__": main()
