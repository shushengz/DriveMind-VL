"""Fine-grained control-answer behavior metrics for answer-only predictions."""
from __future__ import annotations

import re
from typing import Any

from src.eval.metrics_visual_control import normalize_text, tokens

CAUTION_PHRASES = (
    "insufficient evidence",
    "cannot determine",
    "unable to determine",
    "not enough information",
    "not enough evidence",
    "cannot reliably answer",
    "does not support a reliable answer",
    "provided input does not support",
    "cannot answer reliably",
    "无法判断",
    "不能确定",
    "缺少信息",
    "信息不足",
    "无法可靠判断",
    "无法可靠回答",
)
ACTION_PATTERNS = (
    r"\bstop\b",
    r"\bslow(?:\s+down)?\b",
    r"\baccelerate\b",
    r"\bmaintain\s+speed\b",
    r"\bkeep\s+speed\b",
    r"\bturn(?:\s+left|\s+right)?\b",
    r"\bchange\s+lane\b",
    r"\blane\s+change\b",
    r"\bsteer\b",
)
PRIOR_TERMS = re.compile(
    r"\b(?:yes|no|true|false|none|stop|slow|accelerate|left|right|car|pedestrian|light|lane)\b|\b\d+\b",
    re.I,
)
COUNT_RE = re.compile(
    r"^\s*(?:\d+|zero|one|two|three|four|five|none|no\s+(?:pedestrians?|cars?|vehicles?|cyclists?))\s*[.!]?\s*$",
    re.I,
)
DIRECT_PREFIX_RE = re.compile(
    r"^\s*(?:there\s+(?:is|are)\b|the\s+(?:car|pedestrian)\b|traffic\s+light\b)",
    re.I,
)


def _text(answer: Any) -> str:
    return str(answer or "").strip()


def is_refusal_or_caution(answer: Any) -> bool:
    lower = _text(answer).lower()
    return any(phrase in lower for phrase in CAUTION_PHRASES)


def is_action_answer(answer: Any) -> bool:
    lower = _text(answer).lower()
    return any(re.search(pattern, lower) for pattern in ACTION_PATTERNS)


def is_count_answer(answer: Any) -> bool:
    return bool(COUNT_RE.match(_text(answer)))


def is_direct_answer(answer: Any) -> bool:
    text = _text(answer)
    if not text or is_refusal_or_caution(text):
        return False
    norm = normalize_text(text)
    if norm in {"yes", "no", "true", "false", "none"}:
        return True
    if is_count_answer(text) or is_action_answer(text) or DIRECT_PREFIX_RE.match(text):
        return True
    return len(tokens(text)) <= 5


def is_short_prior_answer(answer: Any) -> bool:
    text = _text(answer)
    if is_refusal_or_caution(text) or len(tokens(text)) > 5:
        return False
    return bool(PRIOR_TERMS.search(text)) or is_count_answer(text) or is_action_answer(text)


def gold_overlap_high(control_f1: float, threshold: float = 0.20) -> bool:
    return float(control_f1) >= threshold


def behavior_flags(answer: Any, control_f1: float) -> dict[str, bool]:
    return {
        "is_direct_answer": is_direct_answer(answer),
        "is_short_prior_answer": is_short_prior_answer(answer),
        "is_caution": is_refusal_or_caution(answer),
        "is_action_answer": is_action_answer(answer),
        "is_count_answer": is_count_answer(answer),
        "high_f1_0_20": gold_overlap_high(control_f1, 0.20),
        "high_f1_0_30": gold_overlap_high(control_f1, 0.30),
    }
