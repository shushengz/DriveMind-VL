"""Audit converted external benchmark JSONL files before GPU inference."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.external_vqa_taxonomy import infer_external_vqa_capability


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def row_image_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    image = str(row.get("image") or "")
    if image:
        paths.append(image)
    external = row.get("meta", {}).get("external", {})
    if isinstance(external, dict):
        for item in external.get("image_paths", []) or []:
            path = str(item or "")
            if path:
                paths.append(path)
    return list(dict.fromkeys(paths))


def path_exists(path: str) -> bool:
    if not path:
        return False
    if path.startswith(("http://", "https://")):
        return True
    return Path(path).exists()


def capability(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    return str(
        meta.get("capability")
        or external.get("capability")
        or infer_external_vqa_capability(
            instruction=str(row.get("instruction") or ""),
            category=str(answer.get("category") or external.get("category") or ""),
            subcategory=str(answer.get("subcategory") or external.get("subcategory") or ""),
            reference=str(answer.get("answer") or answer.get("reason") or ""),
        )
    )


def audit(rows: list[dict[str, Any]], max_examples: int) -> dict[str, Any]:
    ids = [str(row.get("id") or "") for row in rows]
    id_counts = Counter(ids)
    missing_id = [index for index, sample_id in enumerate(ids) if not sample_id]
    duplicate_ids = sorted(sample_id for sample_id, count in id_counts.items() if sample_id and count > 1)
    missing_instruction = [str(row.get("id") or index) for index, row in enumerate(rows) if not str(row.get("instruction") or "").strip()]
    missing_answer = []
    missing_media = []
    missing_existing_media = []
    bad_task = []
    by_source = Counter()
    by_capability = Counter()
    by_category = Counter()
    frame_counts: list[int] = []

    for index, row in enumerate(rows):
        sample_id = str(row.get("id") or index)
        answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
        if not str(answer.get("answer") or answer.get("reason") or "").strip():
            missing_answer.append(sample_id)
        task_type = str(row.get("meta", {}).get("task_type") or "")
        if task_type != "external_vqa":
            bad_task.append(sample_id)
        paths = row_image_paths(row)
        video = str(row.get("video") or row.get("meta", {}).get("external", {}).get("video_path", "") or "")
        if not paths and not video:
            missing_media.append(sample_id)
        existing_paths = [path for path in paths if path_exists(path)]
        if paths and not existing_paths:
            missing_existing_media.append(sample_id)
        frame_counts.append(len(paths))
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
        by_source[str(meta.get("benchmark_source") or external.get("benchmark_source") or "unknown")] += 1
        by_capability[capability(row)] += 1
        by_category[str(answer.get("category") or external.get("category") or "unknown")] += 1

    count = len(rows)
    return {
        "count": count,
        "id": {
            "missing": len(missing_id),
            "duplicate_count": len(duplicate_ids),
            "duplicate_examples": duplicate_ids[:max_examples],
        },
        "schema": {
            "missing_instruction": len(missing_instruction),
            "missing_answer": len(missing_answer),
            "non_external_vqa_task": len(bad_task),
            "missing_instruction_examples": missing_instruction[:max_examples],
            "missing_answer_examples": missing_answer[:max_examples],
            "non_external_vqa_examples": bad_task[:max_examples],
        },
        "media": {
            "missing_media": len(missing_media),
            "missing_existing_media": len(missing_existing_media),
            "missing_media_examples": missing_media[:max_examples],
            "missing_existing_media_examples": missing_existing_media[:max_examples],
            "avg_image_paths": round(sum(frame_counts) / count, 4) if count else 0.0,
            "max_image_paths": max(frame_counts) if frame_counts else 0,
        },
        "by_source": dict(by_source),
        "by_capability": dict(by_capability),
        "by_category": dict(by_category.most_common(30)),
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], input_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# External Benchmark Audit",
        "",
        f"Input: `{input_path}`",
        f"Samples: `{report['count']}`",
        "",
        "## Checks",
        "",
        "| check | value |",
        "|---|---:|",
        f"| missing id | {report['id']['missing']} |",
        f"| duplicate id | {report['id']['duplicate_count']} |",
        f"| missing instruction | {report['schema']['missing_instruction']} |",
        f"| missing answer | {report['schema']['missing_answer']} |",
        f"| non external_vqa task | {report['schema']['non_external_vqa_task']} |",
        f"| missing media | {report['media']['missing_media']} |",
        f"| media paths not found | {report['media']['missing_existing_media']} |",
        f"| avg image paths | {report['media']['avg_image_paths']} |",
        "",
        "## By Capability",
        "",
        "| capability | count |",
        "|---|---:|",
    ]
    for name, count in sorted(report["by_capability"].items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## By Source", "", "| source | count |", "|---|---:|"])
    for name, count in sorted(report["by_source"].items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## Top Categories", "", "| category | count |", "|---|---:|"])
    for name, count in report["by_category"].items():
        lines.append(f"| {name} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit converted external benchmark JSONL before inference.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/eval_results/external_benchmark_audit.json")
    parser.add_argument("--markdown", default="docs/external_benchmark_audit.md")
    parser.add_argument("--max_examples", type=int, default=20)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.input))
    report = audit(rows, args.max_examples)
    write_json(Path(args.output), report)
    write_markdown(Path(args.markdown), report, args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
