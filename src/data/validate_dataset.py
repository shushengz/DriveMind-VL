"""Validate DriveMind-Instruct JSONL data."""

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

try:
    from src.agent.tools import get_registered_tools
except Exception:
    from agent.tools import get_registered_tools


VALID_TASKS = {"risk_reasoning", "tool_call", "safety_rejection", "cabin_understanding", "personalized_service"}
VALID_RISK_LEVELS = {"low", "medium", "high"}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as exc:
                yield line_no, {"__error__": f"json_decode_error: {exc}"}


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors = []
    if "__error__" in sample:
        return [sample["__error__"]]
    for field in ("id", "image", "vehicle_state", "perception", "instruction", "answer", "meta"):
        if field not in sample:
            errors.append(f"missing_field:{field}")
    image = sample.get("image")
    if image and not Path(image).exists():
        errors.append("image_not_found")
    answer = sample.get("answer")
    if not isinstance(answer, dict):
        errors.append("answer_not_dict")
        answer = {}
    meta = sample.get("meta")
    if not isinstance(meta, dict):
        errors.append("meta_not_dict")
        meta = {}
    task_type = meta.get("task_type")
    if task_type not in VALID_TASKS:
        errors.append("invalid_task_type")
    if "risk_level" in answer and answer.get("risk_level") not in VALID_RISK_LEVELS:
        errors.append("invalid_risk_level")
    if "tool" in answer and answer.get("tool") not in get_registered_tools():
        errors.append("invalid_tool")
    return errors


def validate_dataset(path: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {"total": 0, "valid": 0, "errors": Counter(), "task_counts": Counter()}
    for line_no, sample in iter_jsonl(path):
        stats["total"] += 1
        errors = validate_sample(sample)
        if errors:
            for error in errors:
                stats["errors"][error] += 1
            print(f"line {line_no}: {', '.join(errors)}")
        else:
            stats["valid"] += 1
            stats["task_counts"][sample["meta"]["task_type"]] += 1
    stats["errors"] = dict(stats["errors"])
    stats["task_counts"] = dict(stats["task_counts"])
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DriveMind-Instruct dataset.")
    parser.add_argument("--input", default="data/processed/drivemind_seed.jsonl")
    args = parser.parse_args()
    stats = validate_dataset(Path(args.input))
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
