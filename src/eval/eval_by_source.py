"""Evaluate DriveMind-VL predictions grouped by data provenance."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_all_eval import collect_bad_cases, compute_metrics, load_jsonl, write_jsonl


def get_source(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    source = (
        meta.get("benchmark_source")
        or external.get("benchmark_source")
        or meta.get("source")
        or "unknown"
    )
    return str(source)


def get_task(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    return str(meta.get("task_type") or gold.get("task") or "unknown")


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if key == "source":
            grouped[get_source(row)].append(row)
        elif key == "task":
            grouped[get_task(row)].append(row)
        elif key == "source_task":
            grouped[f"{get_source(row)}::{get_task(row)}"].append(row)
        else:
            raise ValueError(f"unsupported group key: {key}")
    return dict(grouped)


def build_report(rows: list[dict[str, Any]], bad_case_reward_threshold: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "overall": {"count": len(rows), "metrics": compute_metrics(rows)},
        "by_source": {},
        "by_task": {},
        "by_source_task": {},
    }
    for section, group_key in (
        ("by_source", "source"),
        ("by_task", "task"),
        ("by_source_task", "source_task"),
    ):
        for name, group in sorted(group_rows(rows, group_key).items()):
            bad_cases = collect_bad_cases(group, bad_case_reward_threshold)
            report[section][name] = {
                "count": len(group),
                "bad_cases": len(bad_cases),
                "metrics": compute_metrics(group),
            }
    return report


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_group_table(title: str, groups: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    print("+--------------------------------------+-------+-----------+--------------+-------------+------------+------------+")
    print("| group                                | count | bad_cases | json_validity | avg_reward  | tool_acc   | ext_f1     |")
    print("+--------------------------------------+-------+-----------+--------------+-------------+------------+------------+")
    for name, payload in groups.items():
        metrics = payload["metrics"]
        print(
            f"| {name[:36]:<36} | {payload['count']:<5} | {payload['bad_cases']:<9} | "
            f"{metrics.get('json_validity', 0.0):<12.4f} | {metrics.get('avg_reward', 0.0):<11.4f} | "
            f"{metrics.get('tool_accuracy', 0.0):<10.4f} | {metrics.get('external_answer_f1', 0.0):<10.4f} |"
        )
    print("+--------------------------------------+-------+-----------+--------------+-------------+------------+------------+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriveMind-VL metrics grouped by benchmark source and task.")
    parser.add_argument("--predictions", default="outputs/eval_results/base_predictions.jsonl")
    parser.add_argument("--output", default="outputs/eval_results/metrics_by_source.json")
    parser.add_argument("--bad_cases_output", default="outputs/cases/bad_cases_by_source.jsonl")
    parser.add_argument("--bad_case_reward_threshold", type=float, default=0.65)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions))
    report = build_report(rows, args.bad_case_reward_threshold)
    bad_cases = collect_bad_cases(rows, args.bad_case_reward_threshold)
    write_json(Path(args.output), report)
    write_jsonl(Path(args.bad_cases_output), bad_cases)

    overall = report["overall"]["metrics"]
    print("== Overall ==")
    print(json.dumps({"count": report["overall"]["count"], **overall}, ensure_ascii=False, indent=2))
    print_group_table("By Source", report["by_source"])
    print_group_table("By Task", report["by_task"])
    print(f"\nwrote grouped metrics to {args.output}")
    print(f"wrote {len(bad_cases)} bad cases to {args.bad_cases_output}")


if __name__ == "__main__":
    main()
