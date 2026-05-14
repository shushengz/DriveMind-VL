"""Tool-call reward."""

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


def _args_match(pred_args: Any, gold_args: Any) -> bool:
    if not isinstance(pred_args, dict) or not isinstance(gold_args, dict):
        return False
    for key, value in gold_args.items():
        if pred_args.get(key) != value:
            return False
    return True


def tool_reward(pred: Any, gold: dict[str, Any] | None = None, vehicle_state: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> float:
    try:
        pred_obj = parse_model_output(pred)["data"]
        gold = gold or {}
        score = 0.0
        if pred_obj.get("tool") == gold.get("tool"):
            score += 0.6
        if _args_match(pred_obj.get("arguments"), gold.get("arguments")):
            score += 0.4
        return score
    except Exception:
        return 0.0
