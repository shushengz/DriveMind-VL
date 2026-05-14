"""Robust JSON parser for model outputs."""

from __future__ import annotations

import argparse
import json
import re
from typing import Any


CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_model_output(text_or_obj: Any) -> dict[str, Any]:
    if isinstance(text_or_obj, dict):
        return {"ok": True, "data": text_or_obj, "parse_error": None}
    if text_or_obj is None:
        return {"ok": False, "data": {}, "parse_error": "empty_output"}

    text = str(text_or_obj).strip()
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text))

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return {"ok": True, "data": parsed, "parse_error": None}
            return {"ok": False, "data": {}, "parse_error": "json_not_object"}
        except Exception:
            continue

    return {"ok": False, "data": {}, "parse_error": "json_parse_failed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a DriveMind-VL model output as JSON.")
    parser.add_argument("--text", default='{"task":"risk_reasoning","risk_level":"low"}')
    args = parser.parse_args()
    print(json.dumps(parse_model_output(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

