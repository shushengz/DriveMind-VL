"""Build leakage-safe scene-level splits for converted external VQA data."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def get_external(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    return external


def row_group_key(row: dict[str, Any]) -> str:
    external = get_external(row)
    for key in ("scene_token", "scene_id", "scene", "sample_token", "original_id"):
        value = external.get(key)
        if value not in (None, ""):
            return str(value)
    return str(row.get("id") or "")


def row_capability(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = get_external(row)
    return str(meta.get("capability") or external.get("capability") or "unknown")


def row_category(row: dict[str, Any]) -> str:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    external = get_external(row)
    return str(answer.get("category") or external.get("category") or "unknown")


def grouped_by_scene(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row_group_key(row)
        grouped.setdefault(key, []).append(row)
    return grouped


def choose_dev_scenes(
    groups: dict[str, list[dict[str, Any]]],
    dev_rows: int,
    dev_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    scene_ids = list(groups)
    rng = random.Random(seed)
    rng.shuffle(scene_ids)
    total_rows = sum(len(items) for items in groups.values())
    target_rows = dev_rows if dev_rows > 0 else max(1, round(total_rows * dev_ratio))
    dev: set[str] = set()
    running = 0
    for scene_id in scene_ids:
        if running >= target_rows and dev:
            break
        dev.add(scene_id)
        running += len(groups[scene_id])
    train = set(scene_ids) - dev
    return train, dev


def rows_from_scenes(
    groups: dict[str, list[dict[str, Any]]],
    scene_ids: set[str],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_id in sorted(scene_ids):
        rows.extend(groups[scene_id])
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def clone_for_split(rows: list[dict[str, Any]], source: str, split: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        item = copy.deepcopy(row)
        old_id = str(item.get("id") or f"{index:06d}")
        item["id"] = f"{source}_{split}_{index:06d}"
        meta = item.setdefault("meta", {})
        if isinstance(meta, dict):
            meta["split"] = split
            external = meta.setdefault("external", {})
            if isinstance(external, dict):
                external["split"] = split
                external.setdefault("converted_id", old_id)
        output.append(item)
    return output


def split_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = Counter(row_group_key(row) for row in rows)
    return {
        "count": len(rows),
        "scene_count": len(scenes),
        "avg_rows_per_scene": round(len(rows) / len(scenes), 4) if scenes else 0.0,
        "by_capability": dict(Counter(row_capability(row) for row in rows).most_common()),
        "by_category": dict(Counter(row_category(row) for row in rows).most_common()),
        "scene_examples": list(scenes.keys())[:10],
    }


def build_report(
    input_path: Path,
    source: str,
    seed: int,
    all_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    dev_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    train_scenes = {row_group_key(row) for row in train_rows}
    dev_scenes = {row_group_key(row) for row in dev_rows}
    return {
        "input": input_path.as_posix(),
        "source": source,
        "seed": seed,
        "all": split_stats(all_rows),
        "train": split_stats(train_rows),
        "dev": split_stats(dev_rows),
        "scene_overlap_count": len(train_scenes & dev_scenes),
        "scene_overlap_examples": sorted(train_scenes & dev_scenes)[:20],
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], train_output: str, dev_output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def table(counter: dict[str, int]) -> list[str]:
        lines = ["| name | count |", "|---|---:|"]
        for key, value in counter.items():
            lines.append(f"| {key} | {value} |")
        return lines

    lines = [
        "# External Scene-Level Split Report",
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
    parser = argparse.ArgumentParser(description="Build scene-level train/dev splits for external VQA JSONL.")
    parser.add_argument("--input", required=True, help="Converted DriveMind JSONL from convert_external_to_drivemind.py")
    parser.add_argument("--train_output", default="data/processed/drivelm_train_scene.jsonl")
    parser.add_argument("--dev_output", default="data/processed/drivelm_dev_scene.jsonl")
    parser.add_argument("--report_json", default="outputs/eval_results/drivelm_scene_split_report.json")
    parser.add_argument("--report_md", default="docs/drivelm_scene_split_report.md")
    parser.add_argument("--source", default="drivelm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_rows", type=int, default=600, help="Target dev rows before final cap.")
    parser.add_argument("--dev_ratio", type=float, default=0.15, help="Used only when --dev_rows <= 0.")
    parser.add_argument("--train_limit", type=int, default=2400)
    parser.add_argument("--dev_limit", type=int, default=600)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    if not rows:
        raise SystemExit(f"no rows found in {args.input}")

    groups = grouped_by_scene(rows)
    train_scenes, dev_scenes = choose_dev_scenes(groups, args.dev_rows, args.dev_ratio, args.seed)
    raw_train = rows_from_scenes(groups, train_scenes, args.train_limit, args.seed + 1)
    raw_dev = rows_from_scenes(groups, dev_scenes, args.dev_limit, args.seed + 2)
    train_rows = clone_for_split(raw_train, args.source, "train_scene")
    dev_rows = clone_for_split(raw_dev, args.source, "dev_scene")

    report = build_report(Path(args.input), args.source, args.seed, rows, train_rows, dev_rows)
    if report["scene_overlap_count"] != 0:
        raise RuntimeError(f"scene leakage detected: {report['scene_overlap_examples'][:5]}")

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.dev_output), dev_rows)
    write_json(Path(args.report_json), report)
    write_markdown(Path(args.report_md), report, args.train_output, args.dev_output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {args.train_output}, {args.dev_output}, {args.report_json}, {args.report_md}")


if __name__ == "__main__":
    main()
