"""Prepare a DriveMind external_vqa subset from LingoQA annotations.

The official LingoQA evaluation annotations are stored as a table with at
least question_id, segment_id, question, and answer columns. This script can
read CSV/JSONL/Parquet metadata and does not download videos by default.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.external_vqa_taxonomy import infer_external_vqa_capability, normalize_reference


LINGOQA_EVAL_URL = "https://drive.usercontent.google.com/u/1/uc?id=1I8u6uYysQUstoVYZapyRQkXmOwr-AG3d&export=download"


def load_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [row for row in obj if isinstance(row, dict)]
        for key in ("data", "records", "items"):
            value = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        raise ValueError(f"unsupported JSON structure: {path}")
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("Reading Parquet requires pandas with pyarrow or fastparquet installed.") from exc
        return pd.read_parquet(path).to_dict("records")
    raise ValueError(f"unsupported annotation file extension: {path.suffix}")


def group_references(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    raw_rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        question_id = str(row.get("question_id", ""))
        segment_id = str(row.get("segment_id", ""))
        question = str(row.get("question", ""))
        if not question_id or not segment_id or not question:
            continue
        key = (question_id, segment_id, question)
        answer = normalize_reference(row.get("answer"))
        if answer:
            grouped[key].append(answer)
        raw_rows.setdefault(key, row)

    records = []
    for key, references in grouped.items():
        question_id, segment_id, question = key
        base = dict(raw_rows[key])
        base["question_id"] = question_id
        base["segment_id"] = segment_id
        base["question"] = question
        base["references"] = references
        records.append(base)
    return records


def capability_of(record: dict[str, Any]) -> str:
    return infer_external_vqa_capability(
        instruction=str(record.get("question", "")),
        category=str(record.get("category", "")),
        subcategory=str(record.get("subcategory", "")),
        reference=" ".join(str(ref) for ref in record.get("references", [])),
    )


def balanced_sample(records: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[capability_of(record)].append(record)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    capabilities = sorted(groups)
    cursor = {capability: 0 for capability in capabilities}
    while len(selected) < target_size and capabilities:
        progressed = False
        for capability in capabilities:
            idx = cursor[capability]
            if idx < len(groups[capability]) and len(selected) < target_size:
                selected.append(groups[capability][idx])
                cursor[capability] += 1
                progressed = True
        if not progressed:
            break
    return selected


def maybe_video_path(segment_id: str, video_root: str) -> str:
    if not video_root:
        return ""
    root = Path(video_root)
    candidates = [
        root / f"{segment_id}.mp4",
        root / f"{segment_id}.webm",
        root / f"{segment_id}.mov",
        root / segment_id / "video.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.as_posix()
    return (root / f"{segment_id}.mp4").as_posix()


def normalize_images(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str) and hasattr(value, "__iter__"):
        try:
            return [str(item) for item in value if str(item)]
        except Exception:
            pass
    text = str(value).strip()
    if not text:
        return []
    jpg_paths = re.findall(r"images/[^,\]\s'\";]+?\.jpg", text)
    if jpg_paths:
        return jpg_paths
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed if str(item)]
    return [text]


def resolve_image_paths(record: dict[str, Any], image_root: str) -> list[str]:
    if not image_root:
        return []
    root = Path(image_root)
    resolved = []
    for image in normalize_images(record.get("images")):
        candidate = Path(image)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved.append(candidate.as_posix())
    return resolved


def to_drivemind(record: dict[str, Any], index: int, video_root: str, image_root: str, split: str) -> dict[str, Any]:
    refs = [str(ref) for ref in record.get("references", []) if str(ref)]
    primary_ref = refs[0] if refs else ""
    capability = capability_of(record)
    segment_id = str(record.get("segment_id", ""))
    video_path = maybe_video_path(segment_id, video_root)
    image_paths = resolve_image_paths(record, image_root)
    return {
        "id": f"lingoqa_{split}_{index:06d}",
        "image": image_paths[0] if image_paths else "",
        "video": video_path,
        "vehicle_state": {
            "speed": record.get("speed", 0),
            "weather": record.get("weather", "unknown"),
            "distance_to_front_car": record.get("distance_to_front_car"),
            "yaw_rate": record.get("yaw_rate"),
            "gear": record.get("gear", "unknown"),
            "time": record.get("time", "unknown"),
        },
        "perception": {
            "objects": [],
            "scene": "driving_video",
            "risk_hint": capability,
        },
        "instruction": str(record.get("question", "")),
        "answer": {
            "task": "external_vqa",
            "answer": primary_ref,
            "references": refs,
            "category": "LingoQA",
            "subcategory": capability,
            "reason": primary_ref,
        },
        "meta": {
            "task_type": "external_vqa",
            "source": "external_benchmark",
            "benchmark_source": "lingoqa",
            "difficulty": "unknown",
            "external": {
                "benchmark_source": "lingoqa",
                "split": split,
                "question_id": str(record.get("question_id", "")),
                "segment_id": segment_id,
                "capability": capability,
                "video_path": video_path,
                "image_paths": image_paths,
            },
        },
    }


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[capability_of(record)] += 1
    return {"total": len(records), "capability_counts": dict(sorted(counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LingoQA annotations as DriveMind external_vqa JSONL.")
    parser.add_argument("--input", required=True, help="LingoQA annotation table: parquet/csv/json/jsonl.")
    parser.add_argument("--output", default="data/processed/drivemind_lingoqa_eval_subset.jsonl")
    parser.add_argument("--target_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_root", default="", help="Optional root for LingoQA relative image paths, usually data/external/lingoqa.")
    parser.add_argument("--video_root", default="", help="Optional local video directory for segment_id videos.")
    parser.add_argument("--split", default="eval", choices=["train", "val", "eval", "test"])
    parser.add_argument("--manifest_output", default="outputs/eval_results/lingoqa_eval_subset_manifest.json")
    args = parser.parse_args()

    rows = load_table(Path(args.input))
    grouped = group_references(rows)
    selected = balanced_sample(grouped, target_size=args.target_size, seed=args.seed)
    converted = [
        to_drivemind(record, index=i, video_root=args.video_root, image_root=args.image_root, split=args.split)
        for i, record in enumerate(selected, start=1)
    ]
    write_jsonl(converted, Path(args.output))

    manifest = {
        "input": args.input,
        "target_size": args.target_size,
        "seed": args.seed,
        "source_rows": len(rows),
        "unique_questions": len(grouped),
        "selected": summarize(selected),
        "image_root": args.image_root,
        "video_root": args.video_root,
        "note": "LingoQA is video VQA. Image-root conversion uses extracted key frames when available.",
    }
    Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest_output).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote subset to {args.output}")


if __name__ == "__main__":
    main()
