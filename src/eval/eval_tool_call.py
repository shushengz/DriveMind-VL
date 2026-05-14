"""Tool-call accuracy metric."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.agent.output_parser import parse_model_output
except Exception:
    from agent.output_parser import parse_model_output


def tool_accuracy(rows: list[dict[str, Any]]) -> float:
    tool_rows = [row for row in rows if row.get("meta", {}).get("task_type") in {"tool_call", "personalized_service"}]
    if not tool_rows:
        return 0.0
    correct = 0
    for row in tool_rows:
        pred = parse_model_output(row.get("prediction"))["data"]
        if pred.get("tool") == row.get("gold", {}).get("tool"):
            correct += 1
    return correct / len(tool_rows)
