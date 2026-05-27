
"""Audit SFT-v3-r2 training data before GPU training."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BAD_FIELDS = {"references", "category", "subcategory", "debug"}
CONTROL_TYPES = {"blank_refusal", "wrong_image_caution", "text_only_caution"}
REFUSAL_RE = re.compile(r"无法判断|无法确定|缺少图像信息|图像信息不足|cannot determine|not enough (?:visual )?information|unable to determine", re.I)
PATH_RE = re.compile(r"(?:^|[\s\"'])/?(?:root/|data/|outputs/|checkpoints/|src/)[^\s\"']+", re.I)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def assistant_obj(row: dict[str, Any]) -> tuple[dict[str, str] | None, bool]:
    obj=row.get("assistant")
    if isinstance(obj, dict) and set(obj.keys()) == {"answer", "reason"}:
        return {"answer": str(obj.get("answer", "")), "reason": str(obj.get("reason", ""))}, True
    try:
        msg=(row.get("messages") or [])[-1]
        parsed=json.loads(str(msg.get("content", "")))
        if isinstance(parsed, dict) and set(parsed.keys()) == {"answer", "reason"}:
            return {"answer": str(parsed.get("answer", "")), "reason": str(parsed.get("reason", ""))}, True
    except Exception:
        pass
    return None, False


def has_bad_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(k) in BAD_FIELDS or has_bad_field(v) for k, v in value.items())
    if isinstance(value, list):
        return any(has_bad_field(v) for v in value)
    return False


def length(text: str) -> int:
    return len(text.split()) if re.search(r"[A-Za-z]", text or "") else len(text or "")


def audit(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    total=len(rows)
    type_counts=Counter(str(r.get("sample_type") or "unknown") for r in rows)
    dataset_counts=Counter(str(r.get("dataset") or "unknown") for r in rows)
    valid=0; normal_refusal=0; control_refusal=0; leaks=0
    answer_lengths=[]; reason_lengths=[]; longest_answer=None; longest_reason=None
    samples=defaultdict(list)
    rng=random.Random(seed)
    shuffled=list(rows); rng.shuffle(shuffled)
    for row in rows:
        obj, ok=assistant_obj(row)
        st=str(row.get("sample_type") or "unknown")
        if len(samples[st]) < 5:
            samples[st].append({"id": row.get("id"), "prompt": str(row.get("prompt", ""))[:220], "assistant": obj})
        if not ok or not obj:
            continue
        valid += 1
        answer=obj["answer"]; reason=obj["reason"]
        al=length(answer); rl=length(reason)
        answer_lengths.append(al); reason_lengths.append(rl)
        if longest_answer is None or al > longest_answer["length"]:
            longest_answer={"id": row.get("id"), "length": al, "answer": answer, "sample_type": st}
        if longest_reason is None or rl > longest_reason["length"]:
            longest_reason={"id": row.get("id"), "length": rl, "reason": reason, "sample_type": st}
        text=json.dumps(obj, ensure_ascii=False)
        if has_bad_field(obj) or PATH_RE.search(text):
            leaks += 1
        if st == "normal_visual_qa" and REFUSAL_RE.search(text):
            normal_refusal += 1
        if st in CONTROL_TYPES and REFUSAL_RE.search(answer):
            control_refusal += 1
    normal=type_counts.get("normal_visual_qa",0)
    control=sum(type_counts[t] for t in CONTROL_TYPES)
    spatial=type_counts.get("spatial_hard_negative",0)
    valid_rate=valid/total if total else 0.0
    report={
        "total": total,
        "sample_type_counts": dict(type_counts),
        "sample_type_ratios": {k:v/total for k,v in type_counts.items()} if total else {},
        "dataset_counts": dict(dataset_counts),
        "normal_ratio": normal/total if total else 0.0,
        "control_ratio": control/total if total else 0.0,
        "spatial_ratio": spatial/total if total else 0.0,
        "normal_refusal_rate": normal_refusal/normal if normal else 0.0,
        "control_refusal_rate": control_refusal/control if control else 0.0,
        "output_json_valid_rate": valid_rate,
        "metadata_leak_count": leaks,
        "avg_answer_length": sum(answer_lengths)/len(answer_lengths) if answer_lengths else 0.0,
        "avg_reason_length": sum(reason_lengths)/len(reason_lengths) if reason_lengths else 0.0,
        "longest_answer": longest_answer,
        "longest_reason": longest_reason,
        "samples_by_type": dict(samples),
    }
    passed=(report["normal_ratio"] >= 0.55 and report["control_ratio"] <= 0.35 and report["normal_refusal_rate"] < 0.02 and report["output_json_valid_rate"] > 0.98 and leaks == 0)
    report["train_ready"] = passed
    report["fail_reasons"] = []
    if report["normal_ratio"] < 0.55: report["fail_reasons"].append("normal_visual_qa < 55%")
    if report["control_ratio"] > 0.35: report["fail_reasons"].append("control/refusal > 35%")
    if report["normal_refusal_rate"] >= 0.02: report["fail_reasons"].append("normal refusal rate >= 2%")
    if report["output_json_valid_rate"] <= 0.98: report["fail_reasons"].append("output JSON valid rate <= 98%")
    if leaks: report["fail_reasons"].append("assistant metadata/path leakage")
    return report


def write_md(path: Path, report: dict[str, Any]) -> None:
    lines=["# SFT-v3-r2 数据审计", "", f"总样本数：{report['total']}", "", "## 比例"]
    lines.append(f"- normal_visual_qa?{report['normal_ratio']:.4f}")
    lines.append(f"- control/refusal?{report['control_ratio']:.4f}")
    lines.append(f"- spatial_hard_negative?{report['spatial_ratio']:.4f}")
    lines.append(f"- normal refusal rate?{report['normal_refusal_rate']:.4f}")
    lines.append(f"- control refusal rate?{report['control_refusal_rate']:.4f}")
    lines.append(f"- output JSON valid rate?{report['output_json_valid_rate']:.4f}")
    lines.append(f"- metadata leak count?{report['metadata_leak_count']}")
    lines += ["", "## 样本类型计数"]
    for k,v in sorted(report["sample_type_counts"].items()):
        lines.append(f"- {k}: {v} ({report['sample_type_ratios'].get(k,0):.4f})")
    lines += ["", "## 长度", f"- avg answer length: {report['avg_answer_length']:.2f}", f"- avg reason length: {report['avg_reason_length']:.2f}"]
    lines += ["", "## 训练准入", "通过" if report["train_ready"] else "不通过"]
    if report["fail_reasons"]:
        for item in report["fail_reasons"]:
            lines.append(f"- {item}")
    lines += ["", "## 每类样本预览"]
    for st, samples in sorted(report["samples_by_type"].items()):
        lines.append(f"### {st}")
        for item in samples[:5]:
            lines.append(f"- {item['id']}: {json.dumps(item['assistant'], ensure_ascii=False)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines)+"\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Audit SFT-v3-r2 data.")
    p.add_argument("--input", default="data/train/sft_v3_r2/lingoqa_sft_v3_r2.jsonl")
    p.add_argument("--output_json", default="outputs/data_audit/sft_v3_r2_lingo_audit.json")
    p.add_argument("--output_md", default="outputs/data_audit/sft_v3_r2_lingo_audit.md")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args=parse_args()
    report=audit(read_jsonl(Path(args.input)), args.seed)
    out_json=Path(args.output_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_md(Path(args.output_md), report)
    print(json.dumps({"json": args.output_json, "md": args.output_md, "train_ready": report["train_ready"], "fail_reasons": report["fail_reasons"]}, ensure_ascii=False, indent=2))
    if not report["train_ready"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
