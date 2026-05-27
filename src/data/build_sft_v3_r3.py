
"""Build SFT-v3-r3 answer-only calibration data.

CPU-only: no model loading, inference, or training.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.visual_control_formatter import write_jsonl

SETTINGS=("normal","blank_image","wrong_image","text_only")
CONTROL_TYPES={"blank_calibration","wrong_image_calibration","text_only_calibration"}
NORMAL_TYPES={"normal_replay","spatial_normal_qa","extra_normal_fill"}
SPATIAL_RE=re.compile(r"left|right|front|back|lane|traffic light|pedestrian|vehicle|car|behind|ahead|CAM_FRONT|CAM_BACK", re.I)
REFUSAL_RE=re.compile(r"\u65e0\u6cd5\u5224\u65ad|\u65e0\u6cd5\u786e\u5b9a|\u7f3a\u5c11\u56fe\u50cf\u4fe1\u606f|cannot determine|not enough information|unable to determine", re.I)
FORBIDDEN_CONTROL_RE=re.compile(r"\b(image|visual|visible|scene|frame|picture)\b|\u56fe\u50cf|\u753b\u9762|\u56fe\u4e2d|\u89c6\u89c9", re.I)
PATH_RE=re.compile(r"(?:^|[\s\"'])/?(?:root/|data/|outputs/|checkpoints/|src/)[^\s\"']+", re.I)
BAD_FIELDS={"reason","references","category","subcategory","debug","metadata"}

BLANK_ANSWER="Insufficient evidence to answer reliably."
TEXT_ONLY_ANSWER="Insufficient evidence to answer reliably."
WRONG_ANSWER="The provided input does not support a reliable answer."


def read_jsonl(path: Path, limit:int=0) -> list[dict[str,Any]]:
    rows=[]
    if not path.exists(): return rows
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            if limit and len(rows)>=limit: break
    return rows


def write_json(path: Path, obj: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")


def answer_obj(answer: str) -> dict[str,str]:
    return {"answer": str(answer or "").strip()}


def is_bad_normal_answer(answer: str) -> bool:
    return not str(answer or "").strip() or REFUSAL_RE.search(str(answer or "")) or PATH_RE.search(str(answer or ""))


def make_row(sample: dict[str,Any], sample_type: str, answer: str, setting: str) -> dict[str,Any]:
    sid=str(sample.get("id") or "unknown")
    prompt=str(sample.get("prompt") or sample.get("question") or "")
    assistant=answer_obj(answer)
    return {
        "id": f"{sid}_{sample_type}",
        "dataset": sample.get("dataset", "lingoqa"),
        "sample_type": sample_type,
        "messages": [
            {"role":"user", "content": prompt},
            {"role":"assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
        "image_paths": sample.get("image_paths", []),
        "image_labels": sample.get("image_labels", []),
        "prompt": prompt,
        "assistant": assistant,
        "metadata": {"source_id": sid, "setting": setting, "stage": "sft_v3_r3"},
    }


def read_groups(visual_control_dir: Path, limit:int=0) -> list[dict[str,dict[str,Any]]]:
    maps={s:{} for s in SETTINGS}; order=[]
    for setting in SETTINGS:
        for row in read_jsonl(visual_control_dir / f"lingoqa_strict_{setting}.jsonl"):
            sid=str(row.get("id") or "")
            if not sid: continue
            maps[setting][sid]=row
            if setting=="normal": order.append(sid)
    common=set(maps["normal"])
    for setting in SETTINGS[1:]: common &= set(maps[setting])
    groups=[]
    for sid in order:
        if sid in common:
            groups.append({setting: maps[setting][sid] for setting in SETTINGS})
        if limit and len(groups)>=limit: break
    return groups


def build_candidates(groups: list[dict[str,dict[str,Any]]]) -> dict[str,list[dict[str,Any]]]:
    c=defaultdict(list)
    for group in groups:
        normal=group["normal"]
        gold=str(normal.get("gold") or "").strip()
        if is_bad_normal_answer(gold):
            continue
        question=str(normal.get("question") or normal.get("prompt") or "")
        c["normal_replay"].append(make_row(normal,"normal_replay",gold,"normal"))
        c["extra_normal_fill"].append(make_row(normal,"extra_normal_fill",gold,"normal"))
        if SPATIAL_RE.search(question):
            c["spatial_normal_qa"].append(make_row(normal,"spatial_normal_qa",gold,"normal"))
        c["blank_calibration"].append(make_row(group["blank_image"],"blank_calibration",BLANK_ANSWER,"blank_image"))
        c["wrong_image_calibration"].append(make_row(group["wrong_image"],"wrong_image_calibration",WRONG_ANSWER,"wrong_image"))
        c["text_only_calibration"].append(make_row(group["text_only"],"text_only_calibration",TEXT_ONLY_ANSWER,"text_only"))
    return c


def take(rows:list[dict[str,Any]], count:int, rng:random.Random) -> list[dict[str,Any]]:
    pool=list(rows); rng.shuffle(pool); return pool[:max(0,min(count,len(pool)))]


def sample_rows(c:dict[str,list[dict[str,Any]]], seed:int, max_samples:int=0) -> tuple[list[dict[str,Any]], list[str]]:
    rng=random.Random(seed); warnings=[]
    normal_available=len(c.get("normal_replay",[]))
    if normal_available == 0: raise ValueError("no clean normal_replay candidates")
    target_total=max_samples if max_samples else round(normal_available / 0.70)
    target_total=max(target_total, normal_available)
    counts={
        "normal_replay": min(normal_available, round(target_total*0.70)),
        "spatial_normal_qa": round(target_total*0.10),
        "blank_calibration": round(target_total*0.05),
        "wrong_image_calibration": round(target_total*0.05),
        "text_only_calibration": round(target_total*0.04),
    }
    rows=[]
    for st in ["normal_replay","spatial_normal_qa","blank_calibration","wrong_image_calibration","text_only_calibration"]:
        got=take(c.get(st,[]), counts[st], rng)
        if len(got) < counts[st]: warnings.append(f"{st}: requested {counts[st]}, available {len(got)}")
        rows.extend(got)
    # Fill deficits with extra normal, never extra control.
    if len(rows) < target_total:
        used={r["id"] for r in rows}
        fill=[r for r in c.get("extra_normal_fill",[]) if r["id"] not in used]
        rows.extend(take(fill, target_total-len(rows), rng))
    rng.shuffle(rows)
    return rows, warnings


def parse_assistant(row:dict[str,Any]) -> tuple[dict[str,Any]|None,bool]:
    obj=row.get("assistant")
    if isinstance(obj,dict): return obj, set(obj.keys()) == {"answer"}
    return None, False


def stats(rows:list[dict[str,Any]], warnings:list[str]) -> dict[str,Any]:
    total=len(rows); counts=Counter(str(r.get("sample_type") or "unknown") for r in rows)
    normal_like=sum(counts[t] for t in NORMAL_TYPES); control=sum(counts[t] for t in CONTROL_TYPES)
    valid=0; reason_count=0; normal_refusal=0; forbidden=0; leaks=0; lengths=[]
    for row in rows:
        obj, ok=parse_assistant(row)
        if ok: valid += 1
        if isinstance(obj,dict):
            if "reason" in obj: reason_count += 1
            ans=str(obj.get("answer", ""))
            lengths.append(len(ans.split()) if re.search(r"[A-Za-z]", ans) else len(ans))
            if row.get("sample_type") in NORMAL_TYPES and REFUSAL_RE.search(ans): normal_refusal += 1
            if row.get("sample_type") in CONTROL_TYPES and FORBIDDEN_CONTROL_RE.search(ans): forbidden += 1
            if any(k in BAD_FIELDS for k in obj) or PATH_RE.search(json.dumps(obj,ensure_ascii=False)): leaks += 1
        else:
            leaks += 1
    control_rate=forbidden/control if control else 0.0
    normal_refusal_rate=normal_refusal/normal_like if normal_like else 0.0
    output_valid_rate=valid/total if total else 0.0
    guard=[]
    if total == 0: guard.append("empty dataset")
    if total and normal_like/total < 0.75: guard.append("normal replay + spatial ratio below 0.75")
    if total and control/total > 0.15: guard.append("control ratio above 0.15")
    if normal_refusal_rate >= 0.01: guard.append("normal refusal rate >= 1%")
    if output_valid_rate <= 0.98: guard.append("output JSON valid rate <= 98%")
    if reason_count: guard.append("reason field appears")
    if control_rate >= 0.02: guard.append("control forbidden visual terms rate >= 2%")
    if leaks: guard.append("metadata/path leak in assistant output")
    return {
        "total": total,
        "sample_type_counts": dict(counts),
        "sample_type_ratios": {k:v/total for k,v in counts.items()} if total else {},
        "normal_like_ratio": normal_like/total if total else 0.0,
        "control_ratio": control/total if total else 0.0,
        "output_json_valid_rate": output_valid_rate,
        "avg_answer_length": sum(lengths)/len(lengths) if lengths else 0.0,
        "normal_refusal_rate": normal_refusal_rate,
        "control_forbidden_visual_terms_rate": control_rate,
        "reason_field_count": reason_count,
        "metadata_leak_count": leaks,
        "train_ready": not guard,
        "fail_reasons": guard,
        "warnings": warnings,
    }


def preview_rows(rows:list[dict[str,Any]], per_type:int=5) -> list[dict[str,Any]]:
    out=[]; seen=Counter()
    for row in rows:
        st=str(row.get("sample_type"))
        if seen[st] < per_type:
            out.append(row); seen[st]+=1
    return out


def parse_args():
    p=argparse.ArgumentParser(description="Build SFT-v3-r3 answer-only data.")
    p.add_argument("--input_lingoqa", default="data/processed/visual_control/lingoqa_strict_normal.jsonl")
    p.add_argument("--visual_control_dir", default="data/processed/visual_control")
    p.add_argument("--output_dir", default="data/train/sft_v3_r3")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args=parse_args()
    group_limit=24 if args.dry_run else 0
    max_samples=args.max_samples or (48 if args.dry_run else 0)
    groups=read_groups(Path(args.visual_control_dir), group_limit)
    rows,warnings=sample_rows(build_candidates(groups), args.seed, max_samples)
    st=stats(rows,warnings)
    out=Path(args.output_dir)
    write_jsonl(out/"lingoqa_sft_v3_r3.jsonl", rows)
    write_jsonl(out/"lingoqa_sft_v3_r3_preview.jsonl", preview_rows(rows))
    write_json(out/"lingoqa_sft_v3_r3_stats.json", st)
    print(json.dumps({"output_dir": out.as_posix(), **st}, ensure_ascii=False, indent=2))
    if not st["train_ready"]:
        raise SystemExit(1)

if __name__ == "__main__": main()
