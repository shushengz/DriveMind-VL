"""Depth estimator stub.

TODO: Connect Depth Anything V2 or another monocular depth estimator during the
server/perception phase. The local MVP must not download depth weights.
"""

from __future__ import annotations


def estimate_relative_depth(image_path: str, bbox: list[int] | None = None) -> float:
    return 8.5

