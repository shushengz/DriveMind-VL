"""Build evidence-aware DriveLM SFT rows.

The builder uses DriveLM annotations only to write stronger target reasons.
Training should normally run with ``--perception_mode none`` so these
annotations are not leaked into the prompt.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OBJECT_REF_RE = re.compile(r"<[^>]+>")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def answer_text(answer: dict[str, Any]) -> str:
    return str(answer.get("answer") or answer.get("reason") or "").strip()


def object_camera(object_id: str) -> str:
    parts = object_id.strip("<>").split(",")
    return parts[1] if len(parts) > 1 else ""


def object_phrase(object_id: str, info: dict[str, Any]) -> str:
    category = str(info.get("Category") or info.get("category") or "object").strip()
    status = str(info.get("Status") or info.get("status") or "").strip()
    desc = str(info.get("Visual_description") or info.get("visual_description") or "").strip()
    camera = object_camera(object_id)
    pieces = []
    if desc:
        pieces.append(desc.rstrip("."))
    elif category:
        pieces.append(category)
    if status:
        pieces.append(f"status {status.lower()}")
    if camera:
        pieces.append(f"in {camera}")
    return ", ".join(pieces)


def perception_objects(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    perception = row.get("perception") if isinstance(row.get("perception"), dict) else {}
    objects = perception.get("objects") if isinstance(perception.get("objects"), dict) else {}
    return {str(key): value for key, value in objects.items() if isinstance(value, dict)}


def referenced_evidence(row: dict[str, Any]) -> list[str]:
    objects = perception_objects(row)
    instruction = str(row.get("instruction") or "")
    refs = OBJECT_REF_RE.findall(instruction)
    phrases = [object_phrase(ref, objects[ref]) for ref in refs if ref in objects]
    return [phrase for phrase in phrases if phrase]


def scene_evidence(row: dict[str, Any], max_objects: int) -> list[str]:
    objects = perception_objects(row)
    phrases = [object_phrase(object_id, info) for object_id, info in list(objects.items())[:max_objects]]
    perception = row.get("perception") if isinstance(row.get("perception"), dict) else {}
    scene = str(perception.get("scene") or "").strip()
    if scene and scene != "external_benchmark":
        phrases.insert(0, scene.rstrip("."))
    return [phrase for phrase in phrases if phrase]


def evidence_reason(row: dict[str, Any], max_objects: int) -> tuple[str, str]:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    direct_answer = answer_text(answer)
    referenced = referenced_evidence(row)
    evidence = referenced or scene_evidence(row, max_objects=max_objects)
    if evidence:
        evidence_text = "; ".join(evidence[:max_objects])
        return f"{direct_answer} The visible frames show {evidence_text}.", "object_evidence"
    if direct_answer:
        return (
            f"{direct_answer} The available frames do not provide enough additional visible object evidence beyond the direct answer.",
            "answer_only",
        )
    return "The visual evidence is insufficient to answer this question reliably.", "empty_answer"


def normalize_row(row: dict[str, Any], reason: str, evidence_type: str) -> dict[str, Any]:
    new_row = json.loads(json.dumps(row, ensure_ascii=False))
    answer = new_row.get("answer") if isinstance(new_row.get("answer"), dict) else {}
    if not answer:
        answer = {"task": "external_vqa", "answer": ""}
    answer["task"] = str(answer.get("task") or "external_vqa")
    answer["reason"] = reason
    new_row["answer"] = answer
    meta = new_row.setdefault("meta", {})
    meta["evidence_sft_version"] = "drivelm_evidence_sft_v1"
    meta["evidence_type"] = evidence_type
    meta["recommended_perception_mode"] = "none"
    return new_row


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(Path(args.input))
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    output_rows: list[dict[str, Any]] = []
    by_capability: Counter[str] = Counter()
    by_evidence_type: Counter[str] = Counter()
    for row in rows:
        reason, evidence_type = evidence_reason(row, max_objects=args.max_objects_in_reason)
        new_row = normalize_row(row, reason, evidence_type)
        output_rows.append(new_row)
        meta = new_row.get("meta") if isinstance(new_row.get("meta"), dict) else {}
        by_capability[str(meta.get("capability") or "unknown")] += 1
        by_evidence_type[evidence_type] += 1

    summary = {
        "recipe": "drivelm_evidence_sft_v1",
        "input": args.input,
        "output": args.output,
        "rows": len(output_rows),
        "recommended_training_args": {
            "prompt_variant": "evidence",
            "perception_mode": "none",
            "use_all_images": True,
            "frame_strategy": "uniform",
        },
        "by_capability": dict(by_capability.most_common()),
        "by_evidence_type": dict(by_evidence_type.most_common()),
    }
    return output_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-aware DriveLM SFT data.")
    parser.add_argument("--input", default="data/processed/drivelm_train_scene.jsonl")
    parser.add_argument("--output", default="data/processed/drivelm_evidence_sft_v1_train.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/drivelm_evidence_sft_v1_train_summary.json")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_objects_in_reason", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    rows, summary = build_rows(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output} and {args.summary}")


if __name__ == "__main__":
    main()
