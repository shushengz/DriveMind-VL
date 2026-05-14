"""Format reward for JSON-object outputs."""

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


REQUIRED_BY_TASK = {
    "risk_reasoning": {"task", "risk_level", "risk_object", "reason", "suggestion"},
    "tool_call": {"task", "tool", "arguments", "reason"},
    "safety_rejection": {"task", "refusal", "reason"},
    "cabin_understanding": {"task", "driver_state", "reason", "suggestion"},
    "personalized_service": {"task", "tool", "arguments", "reason"},
}


def format_reward(pred: Any, gold: dict[str, Any] | None = None, vehicle_state: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> float:
    try:
        parsed = parse_model_output(pred)
        if not parsed["ok"]:
            return 0.0
        score = 1.0
        task = (meta or {}).get("task_type") or parsed["data"].get("task")
        required = REQUIRED_BY_TASK.get(task, set())
        if required and required.issubset(parsed["data"].keys()):
            score += 0.2
        return min(score, 1.0)
    except Exception:
        return 0.0
