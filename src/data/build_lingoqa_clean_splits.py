"""Build segment-level clean LingoQA splits and visual-control sets.

This script is CPU-only. It splits the existing DriveMind-formatted LingoQA
500 subset by ``segment_id`` so train/dev/test do not share the same video
clip. It also creates wrong-frame and blank-frame controls for each split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def segment_id(row: dict[str, Any]) -> str:
    return str(row.get("meta", {}).get("external", {}).get("segment_id", ""))


def capability(row: dict[str, Any]) -> str:
    return str(
        row.get("meta", {}).get("external", {}).get("capability")
        or row.get("answer", {}).get("subcategory")
        or row.get("perception", {}).get("risk_hint")
        or "unknown"
    )


def image_paths(row: dict[str, Any]) -> list[str]:
    external = row.get("meta", {}).get("external", {})
    paths = external.get("image_paths")
    if isinstance(paths, list) and paths:
        return [str(path) for path in paths]
    image = row.get("image")
    return [str(image)] if image else []


def ensure_blank(path: Path, size: tuple[int, int] = (640, 360)) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw

        img = Image.new("RGB", size, (128, 128, 128))
        draw = ImageDraw.Draw(img)
        draw.text((24, 24), "blank visual control", fill=(235, 235, 235))
        img.save(path, quality=95)
    except Exception:
        # Fallback PPM bytes with a .jpg extension is not ideal, but this path
        # should rarely be used because Pillow is already needed elsewhere.
        raise RuntimeError("Creating blank control image requires Pillow.")


def assign_segments(rows: list[dict[str, Any]], train_ratio: float, dev_ratio: float, seed: int) -> dict[str, str]:
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sid = segment_id(row)
        if sid:
            by_segment[sid].append(row)
    segments = sorted(by_segment)
    random.Random(seed).shuffle(segments)
    train_n = round(len(segments) * train_ratio)
    dev_n = round(len(segments) * dev_ratio)
    split_by_segment: dict[str, str] = {}
    for idx, sid in enumerate(segments):
        if idx < train_n:
            split_by_segment[sid] = "train"
        elif idx < train_n + dev_n:
            split_by_segment[sid] = "dev"
        else:
            split_by_segment[sid] = "test"
    return split_by_segment


def with_split_meta(row: dict[str, Any], split: str) -> dict[str, Any]:
    out = copy.deepcopy(row)
    out.setdefault("meta", {})
    out["meta"]["clean_split"] = split
    out["meta"]["source"] = "lingoqa_clean_v2"
    out["meta"]["benchmark_source"] = "lingoqa"
    return out


def make_wrong_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_segment[segment_id(row)].append(row)
    segments = sorted(by_segment)
    if len(segments) < 2:
        raise ValueError("Need at least two segments to build wrong-frame controls.")
    wrong_segment = {sid: segments[(idx + max(1, len(segments) // 2)) % len(segments)] for idx, sid in enumerate(segments)}

    wrong_rows: list[dict[str, Any]] = []
    for row in rows:
        source_sid = segment_id(row)
        donor = by_segment[wrong_segment[source_sid]][0]
        donor_paths = image_paths(donor)
        out = copy.deepcopy(row)
        out["image"] = donor_paths[0] if donor_paths else ""
        out.setdefault("meta", {}).setdefault("external", {})
        external = out["meta"]["external"]
        external["original_image"] = row.get("image", "")
        external["original_image_paths"] = image_paths(row)
        external["ablated_image"] = out["image"]
        external["ablated_image_paths"] = donor_paths
        external["image_paths"] = donor_paths
        external["wrong_segment_id"] = segment_id(donor)
        out["meta"]["visual_ablation"] = "wrong_image"
        wrong_rows.append(out)
    return wrong_rows


def make_blank_rows(rows: list[dict[str, Any]], blank_path: str) -> list[dict[str, Any]]:
    blank_rows: list[dict[str, Any]] = []
    for row in rows:
        count = max(1, len(image_paths(row)))
        blank_paths = [blank_path] * count
        out = copy.deepcopy(row)
        out["image"] = blank_path
        out.setdefault("meta", {}).setdefault("external", {})
        external = out["meta"]["external"]
        external["original_image"] = row.get("image", "")
        external["original_image_paths"] = image_paths(row)
        external["ablated_image"] = blank_path
        external["ablated_image_paths"] = blank_paths
        external["image_paths"] = blank_paths
        out["meta"]["visual_ablation"] = "blank_image"
        blank_rows.append(out)
    return blank_rows


def summarize(rows_by_split: dict[str, list[dict[str, Any]]], split_by_segment: dict[str, str]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "segment_counts": dict(Counter(split_by_segment.values())),
        "splits": {},
    }
    for split, rows in rows_by_split.items():
        summary["splits"][split] = {
            "rows": len(rows),
            "segments": len({segment_id(row) for row in rows}),
            "by_capability": dict(Counter(capability(row) for row in rows)),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean LingoQA train/dev/test splits and controls.")
    parser.add_argument("--input", default="data/processed/drivemind_lingoqa_eval_500.jsonl")
    parser.add_argument("--prefix", default="lingoqa_clean_v2")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--dev_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--blank_image", default="outputs/cases/ablation_images/lingoqa_blank.jpg")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_clean_v2_split_summary.json")
    parser.add_argument("--assignment_csv", default="outputs/cases/lingoqa_clean_v2_segment_splits.csv")
    args = parser.parse_args()

    rows = read_jsonl(ROOT / args.input)
    split_by_segment = assign_segments(rows, args.train_ratio, args.dev_ratio, args.seed)
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": []}
    for row in rows:
        split = split_by_segment[segment_id(row)]
        rows_by_split[split].append(with_split_meta(row, split))

    blank_rel = args.blank_image
    ensure_blank(ROOT / blank_rel)

    for split, split_rows in rows_by_split.items():
        write_jsonl(ROOT / f"data/processed/{args.prefix}_{split}.jsonl", split_rows)
        write_jsonl(ROOT / f"data/processed/{args.prefix}_{split}_wrong_frame.jsonl", make_wrong_rows(split_rows))
        write_jsonl(ROOT / f"data/processed/{args.prefix}_{split}_blank_frame.jsonl", make_blank_rows(split_rows, blank_rel))

    assignment_path = ROOT / args.assignment_csv
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["segment_id", "split", "rows", "capabilities"])
        writer.writeheader()
        rows_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            rows_by_segment[segment_id(row)].append(row)
        for sid in sorted(rows_by_segment):
            writer.writerow(
                {
                    "segment_id": sid,
                    "split": split_by_segment[sid],
                    "rows": len(rows_by_segment[sid]),
                    "capabilities": json.dumps(dict(Counter(capability(row) for row in rows_by_segment[sid])), ensure_ascii=False),
                }
            )

    summary = summarize(rows_by_split, split_by_segment)
    summary["input"] = args.input
    summary["prefix"] = args.prefix
    summary["blank_image"] = blank_rel
    summary["assignment_csv"] = args.assignment_csv
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
