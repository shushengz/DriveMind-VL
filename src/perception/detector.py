"""Detector stub.

TODO: Connect YOLO, GroundingDINO, or another detector during the perception
enhancement phase. The local MVP must not download detector weights.
"""

from __future__ import annotations

from typing import Any


def detect_objects(image_path: str) -> list[dict[str, Any]]:
    return [
        {
            "class": "car",
            "bbox": [320, 210, 480, 360],
            "position": "front",
            "confidence": 0.5,
        }
    ]

