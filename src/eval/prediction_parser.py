
"""Prediction parser for answer-only rescoring.

The parser is intentionally permissive: raw predictions are never modified on
 disk, and malformed generations fall back to plain text scoring.
"""
from __future__ import annotations

import ast
import json
import re
from typing import Any

FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.I | re.S)


def _strip_fence(text: str) -> str:
    value = str(text or "").strip()
    match = FENCE_RE.match(value)
    if match:
        return match.group(1).strip()
    return value


def _as_result(raw: str, answer: str, reason: str, parse_success: bool, format_type: str) -> dict[str, Any]:
    return {
        "raw_prediction": raw,
        "answer_text": str(answer or "").strip(),
        "reason_text": str(reason or "").strip(),
        "parse_success": bool(parse_success),
        "format_type": format_type,
    }


def _from_mapping(raw: str, obj: dict[str, Any], format_type: str) -> dict[str, Any] | None:
    if "answer" not in obj and "prediction" not in obj:
        return None
    answer = obj.get("answer", obj.get("prediction", ""))
    reason = obj.get("reason", "")
    return _as_result(raw, str(answer), str(reason), True, format_type)


def _json_like_parse(text: str) -> dict[str, Any] | None:
    candidates = [text]
    # Extract the largest object-like span when extra prose surrounds JSON.
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            obj = ast.literal_eval(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        fixed = candidate
        fixed = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', fixed)
        fixed = fixed.replace("'", '"')
        fixed = re.sub(r",\s*}", "}", fixed)
        try:
            obj = json.loads(fixed)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # Last-resort field extraction for answer/reason-ish outputs.
    answer_match = re.search(r'"?answer"?\s*[:=]\s*"?([^"}\n]+)', text, re.I)
    reason_match = re.search(r'"?reason"?\s*[:=]\s*"?([^"}\n]+)', text, re.I)
    if answer_match:
        answer = answer_match.group(1).strip().rstrip(",")
        if not answer or answer in {":", "=", ","}:
            return None
        return {"answer": answer, "reason": reason_match.group(1).strip().rstrip(",") if reason_match else ""}
    return None


def parse_prediction(prediction: Any) -> dict[str, Any]:
    raw = "" if prediction is None else str(prediction)
    text = _strip_fence(raw)
    if not text:
        return _as_result(raw, "", "", False, "failed")
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            parsed = _from_mapping(raw, obj, "json")
            if parsed:
                return parsed
        if isinstance(obj, str):
            return _as_result(raw, obj, "", True, "json")
    except Exception:
        pass
    obj = _json_like_parse(text)
    if isinstance(obj, dict):
        parsed = _from_mapping(raw, obj, "json_like")
        if parsed:
            return parsed
    # Plain text is parse-successful as text, except object-looking failed JSON.
    if text.lstrip().startswith("{"):
        return _as_result(raw, raw, "", False, "failed")
    return _as_result(raw, text, "", True, "plain_text")
