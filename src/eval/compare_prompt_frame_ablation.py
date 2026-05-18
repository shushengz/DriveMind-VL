"""Summarize LingoQA prompt/frame ablation metrics."""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any


TAG_RE = re.compile(r"prompt-(?P<prompt>[^_]+)_frame-(?P<frame>.+?)_(?P<images>\d+)img")


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"skip {path}: {exc}")
        return {}


def parse_run_name(path: Path) -> dict[str, str]:
    match = TAG_RE.search(path.stem)
    if not match:
        return {"prompt_variant": "unknown", "frame_strategy": "unknown", "max_images": "unknown"}
    return {
        "prompt_variant": match.group("prompt"),
        "frame_strategy": match.group("frame"),
        "max_images": match.group("images"),
    }


def find_breakdown(metrics_path: Path, breakdowns: dict[str, Path]) -> dict[str, Any]:
    key = metrics_path.name.replace("_metrics.json", "_breakdown.json")
    path = breakdowns.get(key)
    return read_json(path) if path else {}


def build_rows(metrics_glob: str, breakdown_glob: str) -> list[dict[str, Any]]:
    breakdowns = {Path(path).name: Path(path) for path in glob.glob(breakdown_glob)}
    rows: list[dict[str, Any]] = []
    for path_str in sorted(glob.glob(metrics_glob)):
        path = Path(path_str)
        metrics = read_json(path)
        breakdown = find_breakdown(path, breakdowns)
        overall = breakdown.get("overall", {}) if isinstance(breakdown.get("overall"), dict) else {}
        row = {
            **parse_run_name(path),
            "metrics_file": path.as_posix(),
            "json_validity": overall.get("json_validity", metrics.get("json_validity", 0.0)),
            "answer_f1": overall.get("external_answer_f1", metrics.get("external_answer_f1", 0.0)),
            "pass_rate": overall.get("pass_rate", metrics.get("pass_rate", 0.0)),
            "avg_reward": overall.get("avg_reward", metrics.get("avg_reward", 0.0)),
            "by_capability": breakdown.get("by_capability", {}),
        }
        rows.append(row)
    rows.sort(key=lambda item: (float(item.get("answer_f1", 0.0)), float(item.get("avg_reward", 0.0))), reverse=True)
    return rows


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_from_capability(row: dict[str, Any], capability: str, metric: str = "external_answer_f1") -> float:
    cap = row.get("by_capability", {}).get(capability, {})
    try:
        return float(cap.get(metric, 0.0))
    except Exception:
        return 0.0


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LingoQA Prompt / Frame Ablation Report",
        "",
        "## Setup",
        "",
        "This report compares prompt variants and frame-selection strategies on the 100-sample LingoQA control subset.",
        "It is intended as a lightweight server experiment before any additional SFT/RFT training.",
        "",
        "## Overall",
        "",
        "| prompt | frame | images | answer_f1 | pass_rate | avg_reward |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['prompt_variant']} | {row['frame_strategy']} | {row['max_images']} | "
            f"{float(row['answer_f1']):.4f} | {float(row['pass_rate']):.4f} | {float(row['avg_reward']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Capability Focus",
            "",
            "| prompt | frame | spatial_f1 | object_f1 | counting_f1 | reasoning_f1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['prompt_variant']} | {row['frame_strategy']} | "
            f"{metric_from_capability(row, 'spatial_localization'):.4f} | "
            f"{metric_from_capability(row, 'object_recognition'):.4f} | "
            f"{metric_from_capability(row, 'counting'):.4f} | "
            f"{metric_from_capability(row, 'reasoning_world_knowledge'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## How To Read",
            "",
            "- A useful prompt/frame strategy should improve spatial localization without collapsing object recognition.",
            "- Overall F1 alone is not enough because LingoQA is skewed toward spatial localization.",
            "- If a strategy improves text-only-like answers but not visual-control gaps, it should not be treated as grounding progress.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def print_table(rows: list[dict[str, Any]]) -> None:
    print("+------------+-------------------+--------+----------+----------+------------+")
    print("| prompt     | frame             | images | f1       | pass     | reward     |")
    print("+------------+-------------------+--------+----------+----------+------------+")
    for row in rows:
        print(
            f"| {row['prompt_variant'][:10]:<10} | {row['frame_strategy'][:17]:<17} | "
            f"{row['max_images']:<6} | {float(row['answer_f1']):<8.4f} | "
            f"{float(row['pass_rate']):<8.4f} | {float(row['avg_reward']):<10.4f} |"
        )
    print("+------------+-------------------+--------+----------+----------+------------+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prompt/frame ablation metrics.")
    parser.add_argument("--metrics_glob", required=True)
    parser.add_argument("--breakdown_glob", required=True)
    parser.add_argument("--output", default="outputs/eval_results/lingoqa_prompt_frame_ablation_summary.json")
    parser.add_argument("--report_output", default="docs/lingoqa_prompt_frame_ablation_report.md")
    args = parser.parse_args()

    rows = build_rows(args.metrics_glob, args.breakdown_glob)
    write_json(Path(args.output), rows)
    write_report(Path(args.report_output), rows)
    print_table(rows)
    print(f"wrote ablation summary to {args.output}")
    print(f"wrote ablation report to {args.report_output}")


if __name__ == "__main__":
    main()
