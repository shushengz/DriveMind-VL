"""Build scene-level DriveLM splits directly from the raw nested JSON file.

This avoids the memory-heavy path of first flattening the entire DriveLM file
into one huge converted JSONL before selecting train/dev subsets.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.convert_external_to_drivemind import convert_record, first_present
from src.data.external_vqa_taxonomy import infer_external_vqa_capability


def load_raw(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("DriveLM raw file must be a scene-token dictionary")
    return obj


def scene_qa_count(scene: dict[str, Any]) -> int:
    total = 0
    key_frames = scene.get("key_frames", {})
    if not isinstance(key_frames, dict):
        return 0
    for frame in key_frames.values():
        if not isinstance(frame, dict):
            continue
        qa_by_task = frame.get("QA", {})
        if not isinstance(qa_by_task, dict):
            continue
        for qa_items in qa_by_task.values():
            if isinstance(qa_items, list):
                total += sum(1 for item in qa_items if isinstance(item, dict))
    return total


def choose_dev_scenes(
    scene_counts: dict[str, int],
    dev_rows: int,
    dev_ratio: float,
    seed: int,
    min_dev_scenes: int,
) -> tuple[list[str], list[str]]:
    scene_tokens = [token for token, count in scene_counts.items() if count > 0]
    rng = random.Random(seed)
    rng.shuffle(scene_tokens)
    total = sum(scene_counts[token] for token in scene_tokens)
    target = dev_rows if dev_rows > 0 else max(1, round(total * dev_ratio))
    dev: list[str] = []
    running = 0
    for token in scene_tokens:
        if running >= target and len(dev) >= min_dev_scenes:
            break
        dev.append(token)
        running += scene_counts[token]
    dev_set = set(dev)
    train = [token for token in scene_tokens if token not in dev_set]
    return train, dev


def iter_raw_records(raw: dict[str, Any], scene_tokens: Iterable[str]) -> Iterable[dict[str, Any]]:
    for scene_token in scene_tokens:
        scene = raw.get(scene_token, {})
        if not isinstance(scene, dict):
            continue
        scene_description = scene.get("scene_description", "")
        key_frames = scene.get("key_frames", {})
        if not isinstance(key_frames, dict):
            continue
        for frame_token, frame in key_frames.items():
            if not isinstance(frame, dict):
                continue
            image_paths = frame.get("image_paths", {})
            key_object_infos = frame.get("key_object_infos", {})
            qa_by_task = frame.get("QA", {})
            if not isinstance(qa_by_task, dict):
                continue
            for task_name, qa_items in qa_by_task.items():
                if not isinstance(qa_items, list):
                    continue
                for qa_index, qa in enumerate(qa_items, start=1):
                    if not isinstance(qa, dict):
                        continue
                    question = first_present(qa, ["Q", "question", "query"], "")
                    answer = first_present(qa, ["A", "answer", "gt_answer"], "")
                    if not question and not answer:
                        continue
                    yield {
                        "id": f"{scene_token}_{frame_token}_{task_name}_{qa_index:03d}",
                        "scene_token": scene_token,
                        "sample_token": frame_token,
                        "question": question,
                        "answer": answer,
                        "category": task_name,
                        "subcategory": first_present(qa, ["layer", "cluster"], ""),
                        "context": first_present(qa, ["C", "context"], ""),
                        "con_up": qa.get("con_up"),
                        "con_down": qa.get("con_down"),
                        "layer": qa.get("layer"),
                        "cluster": qa.get("cluster"),
                        "scene": scene_description,
                        "scene_description": scene_description,
                        "image_paths": image_paths,
                        "objects": key_object_infos,
                        "raw_qa": qa,
                    }


def raw_capability(record: dict[str, Any]) -> str:
    return infer_external_vqa_capability(
        instruction=str(record.get("question") or ""),
        category=str(record.get("category") or ""),
        subcategory=str(record.get("subcategory") or ""),
        reference=str(record.get("answer") or ""),
    )


def collect_split_rows(
    raw: dict[str, Any],
    scene_tokens: list[str],
    source: str,
    split: str,
    image_root: Path | None,
    limit: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    shuffled_scenes = list(scene_tokens)
    rng.shuffle(shuffled_scenes)
    raw_records = list(iter_raw_records(raw, shuffled_scenes))
    rng.shuffle(raw_records)
    if limit and limit > 0:
        raw_records = raw_records[:limit]

    rows: list[dict[str, Any]] = []
    capability_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    for index, record in enumerate(raw_records, start=1):
        capability_counts[raw_capability(record)] += 1
        category_counts[str(record.get("category") or "unknown")] += 1
        scene_counts[str(record.get("scene_token") or "unknown")] += 1
        rows.append(convert_record(record, index=index, source=source, image_root=image_root, split=split))

    stats = {
        "count": len(rows),
        "scene_count": len(scene_counts),
        "avg_rows_per_scene": round(len(rows) / len(scene_counts), 4) if scene_counts else 0.0,
        "by_capability": dict(capability_counts.most_common()),
        "by_category": dict(category_counts.most_common()),
        "scene_examples": list(scene_counts.keys())[:10],
    }
    return rows, stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], train_output: str, dev_output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def table(items: dict[str, int]) -> list[str]:
        lines = ["| name | count |", "|---|---:|"]
        for key, value in items.items():
            lines.append(f"| {key} | {value} |")
        return lines

    lines = [
        "# DriveLM Scene-Level Split Report",
        "",
        f"Input: `{report['input']}`",
        f"Train: `{train_output}`",
        f"Dev: `{dev_output}`",
        f"Seed: `{report['seed']}`",
        "",
        "## Leakage Check",
        "",
        f"Scene overlap count: `{report['scene_overlap_count']}`",
        "",
        "## Counts",
        "",
        "| split | rows | scenes | avg rows / scene |",
        "|---|---:|---:|---:|",
    ]
    for split in ("all", "train", "dev"):
        stats = report[split]
        lines.append(
            f"| {split} | {stats['count']} | {stats['scene_count']} | {stats['avg_rows_per_scene']} |"
        )
    for split in ("train", "dev"):
        lines.extend(["", f"## {split.title()} By Capability", ""])
        lines.extend(table(report[split]["by_capability"]))
        lines.extend(["", f"## {split.title()} By Category", ""])
        lines.extend(table(report[split]["by_category"]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM scene-level train/dev splits from raw nested JSON.")
    parser.add_argument("--input", default="data/external/drivelm/v1_1_train_nus.json")
    parser.add_argument("--image_root", default="data/external/drivelm")
    parser.add_argument("--train_output", default="data/processed/drivelm_train_scene.jsonl")
    parser.add_argument("--dev_output", default="data/processed/drivelm_dev_scene.jsonl")
    parser.add_argument("--report_json", default="outputs/eval_results/drivelm_scene_split_report.json")
    parser.add_argument("--report_md", default="docs/drivelm_scene_split_report.md")
    parser.add_argument("--source", default="drivelm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_rows", type=int, default=600)
    parser.add_argument("--dev_ratio", type=float, default=0.15)
    parser.add_argument("--min_dev_scenes", type=int, default=30)
    parser.add_argument("--train_limit", type=int, default=2400)
    parser.add_argument("--dev_limit", type=int, default=600)
    args = parser.parse_args()

    raw = load_raw(Path(args.input))
    scene_counts = {token: scene_qa_count(scene) for token, scene in raw.items() if isinstance(scene, dict)}
    train_scenes, dev_scenes = choose_dev_scenes(
        scene_counts,
        args.dev_rows,
        args.dev_ratio,
        args.seed,
        args.min_dev_scenes,
    )
    image_root = Path(args.image_root) if args.image_root else None

    train_rows, train_stats = collect_split_rows(
        raw,
        train_scenes,
        source=args.source,
        split="train_scene",
        image_root=image_root,
        limit=args.train_limit,
        seed=args.seed + 1,
    )
    dev_rows, dev_stats = collect_split_rows(
        raw,
        dev_scenes,
        source=args.source,
        split="dev_scene",
        image_root=image_root,
        limit=args.dev_limit,
        seed=args.seed + 2,
    )

    overlap = set(train_scenes) & set(dev_scenes)
    report = {
        "input": Path(args.input).as_posix(),
        "source": args.source,
        "seed": args.seed,
        "all": {
            "count": sum(scene_counts.values()),
            "scene_count": len(scene_counts),
            "avg_rows_per_scene": round(sum(scene_counts.values()) / len(scene_counts), 4) if scene_counts else 0.0,
            "by_capability": {},
            "by_category": {},
            "scene_examples": list(scene_counts.keys())[:10],
        },
        "train": train_stats,
        "dev": dev_stats,
        "scene_overlap_count": len(overlap),
        "scene_overlap_examples": sorted(overlap)[:20],
        "selected_scene_counts": {
            "train": len(train_scenes),
            "dev": len(dev_scenes),
        },
    }
    if overlap:
        raise RuntimeError(f"scene leakage detected: {sorted(overlap)[:5]}")

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.dev_output), dev_rows)
    write_json(Path(args.report_json), report)
    write_markdown(Path(args.report_md), report, args.train_output, args.dev_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {args.train_output}, {args.dev_output}, {args.report_json}, {args.report_md}")


if __name__ == "__main__":
    main()
