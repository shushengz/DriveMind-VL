"""Export DriveMind LingoQA predictions to the official CSV layout."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.output_parser import parse_model_output


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"skip invalid JSONL line {line_no}: {exc}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def prediction_text(row: dict[str, Any]) -> str:
    parsed = parse_model_output(row.get("prediction"))
    if parsed.get("ok") and isinstance(parsed.get("data"), dict):
        data = parsed["data"]
        return str(data.get("answer") or data.get("reason") or "")
    return str(row.get("prediction") or "")


def export_csv(rows: list[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question_id", "segment_id", "answer"])
        writer.writeheader()
        for row in rows:
            external = row.get("meta", {}).get("external", {})
            question_id = str(external.get("question_id") or "")
            segment_id = str(external.get("segment_id") or "")
            if not question_id or not segment_id:
                continue
            writer.writerow(
                {
                    "question_id": question_id,
                    "segment_id": segment_id,
                    "answer": prediction_text(row),
                }
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DriveMind LingoQA prediction JSONL to official CSV columns.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions))
    count = export_csv(rows, Path(args.output))
    print(f"wrote {count} LingoQA prediction rows to {args.output}")


if __name__ == "__main__":
    main()
