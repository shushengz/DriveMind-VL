"""Risk task reward."""

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


def risk_reward(pred: Any, gold: dict[str, Any] | None = None, vehicle_state: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> float:
    try:
        pred_obj = parse_model_output(pred)["data"]
        gold = gold or {}
        score = 0.0
        if pred_obj.get("risk_level") == gold.get("risk_level"):
            score += 0.6
        if pred_obj.get("risk_object") == gold.get("risk_object"):
            score += 0.2
        if pred_obj.get("suggestion") == gold.get("suggestion"):
            score += 0.2
        return score
    except Exception:
        return 0.0
