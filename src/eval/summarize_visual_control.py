"""Summarize existing raw visual-control predictions without inference."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics_visual_control import summarize_rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_rows(path: Path) -> list[dict[str, Any]]:
    return read_csv(path) if path.suffix.lower() == ".csv" else read_jsonl(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "unknown").strip("_") or "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize raw visual-control predictions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model_name", default="")
    parser.add_argument("--mode", default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input))
    if args.dry_run:
        ids = []
        for row in rows:
            sid = row.get("id")
            if sid not in ids:
                ids.append(sid)
            if len(ids) >= 8:
                break
        keep_ids = set(ids)
        rows = [row for row in rows if row.get("id") in keep_ids]
    if not rows:
        raise ValueError("no prediction rows found")
    summary, cases = summarize_rows(rows, seed=args.seed)
    first = rows[0]
    dataset = slug(args.dataset or str(first.get("dataset") or "unknown"))
    model = slug(args.model_name or str(first.get("model_name") or "unknown_model"))
    mode = slug(args.mode or str(first.get("mode") or "strict_visual"))
    out_root = Path(args.output_dir)
    summary_path = out_root / "eval_results" / f"{dataset}_{model}_{mode}_summary.csv"
    cases_path = out_root / "cases" / f"{dataset}_{model}_{mode}_case_scores.csv"
    write_csv(summary_path, [{"dataset": dataset, "model_name": model, "mode": mode, **summary}])
    write_csv(cases_path, cases)
    print(json.dumps({"summary": summary_path.as_posix(), "case_scores": cases_path.as_posix(), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
