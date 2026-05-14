"""Build lightweight perception JSON for the local MVP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PERCEPTION = {
    "objects": [
        {
            "class": "car",
            "bbox": [320, 210, 480, 360],
            "position": "front",
            "relative_depth": 8.5,
            "risk_score": 0.83,
        }
    ],
    "scene": "urban_road",
    "risk_hint": "front_car_close",
}


def build_perception(image_path: str | None = None, existing: dict[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(existing, dict) and existing:
        return existing
    if image_path and "cabin" in Path(image_path).name:
        return {
            "objects": [{"class": "driver_face", "bbox": [250, 90, 390, 260], "position": "front_left", "relative_depth": 1.0, "risk_score": 0.35}],
            "scene": "cabin",
            "risk_hint": "normal_cabin",
        }
    return dict(DEFAULT_PERCEPTION)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lightweight perception JSON.")
    parser.add_argument("--image", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = build_perception(args.image)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

