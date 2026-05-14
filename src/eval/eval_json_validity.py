"""JSON validity metric."""

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


def json_validity(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    ok = sum(1 for row in rows if parse_model_output(row.get("prediction"))["ok"])
    return ok / len(rows)
