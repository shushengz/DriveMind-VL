"""Small refusal detector used by LingoQA control evaluations.

The detector is intentionally conservative: it only marks common explicit
uncertainty / refusal wording, so normal short answers are not mislabeled.
"""

from __future__ import annotations

import json
import re
from typing import Any


REFUSAL_PATTERNS = [
    r"\bcannot\s+(?:determine|answer|tell|infer|identify|see|verify)\b",
    r"\bcan't\s+(?:determine|answer|tell|infer|identify|see|verify)\b",
    r"\bunable\s+to\s+(?:determine|answer|tell|infer|identify|see|verify)\b",
    r"\bnot\s+enough\s+(?:visual\s+)?(?:information|evidence)\b",
    r"\binsufficient\s+(?:visual\s+)?(?:information|evidence)\b",
    r"\bno\s+(?:image|visual|frame|scene)\s+(?:input|evidence|information)\b",
    r"\bblank\s+(?:image|frame|frames)\b",
    r"\bnot\s+visible\b",
    r"\bdo\s+not\s+provide\s+reliable\s+evidence\b",
    r"\bwould\s+be\s+unreliable\b",
    r"\bshould\s+not\s+guess\b",
]


def _maybe_json(value: str) -> Any:
    value = value.strip()
    if not value:
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def prediction_text(prediction: Any) -> str:
    """Return text fields that matter for refusal detection."""

    if isinstance(prediction, str):
        prediction = _maybe_json(prediction)
    if isinstance(prediction, dict):
        parts = []
        for key in ("answer", "reason", "refusal", "safe_alternative"):
            value = prediction.get(key)
            if value is not None:
                parts.append(str(value))
        return " ".join(parts)
    return str(prediction or "")


def is_refusal_text(text: str) -> bool:
    normalized = " ".join(str(text or "").lower().split())
    if not normalized:
        return False
    return any(re.search(pattern, normalized) for pattern in REFUSAL_PATTERNS)


def is_refusal_prediction(prediction: Any) -> bool:
    return is_refusal_text(prediction_text(prediction))


def refusal_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    count = len(rows)
    refusals = sum(1 for row in rows if is_refusal_prediction(row.get("prediction")))
    return {
        "count": count,
        "refusal_count": refusals,
        "refusal_rate": round(refusals / count, 4) if count else 0.0,
        "answer_rate": round((count - refusals) / count, 4) if count else 0.0,
    }
