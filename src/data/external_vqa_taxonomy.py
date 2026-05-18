"""Shared taxonomy helpers for external VQA data."""

from __future__ import annotations

from typing import Any


def build_text_blob(*values: Any) -> str:
    return " ".join(str(value).lower() for value in values if value not in (None, ""))


def infer_external_vqa_capability(
    instruction: str = "",
    category: str = "",
    subcategory: str = "",
    reference: str = "",
) -> str:
    blob = build_text_blob(instruction, category, subcategory, reference)
    if any(term in blob for term in ("quantitative", "how many", "count", "number of")):
        return "counting"
    if any(term in blob for term in ("object_recognition", "recognition", "brand", "type recognition", "license", "color", "what brand")):
        return "object_recognition"
    if any(term in blob for term in ("weather_road_condition", "weather", "rain", "fog", "wet", "reflective", "road condition", "environmental")):
        return "weather_road_condition"
    if any(term in blob for term in ("spatial_localization", "spatial", "localization", "left", "right", "front", "rear", "ahead", "behind", "side", "where")):
        return "spatial_localization"
    if any(term in blob for term in ("description", "what's ahead", "what is ahead", "what can you see")):
        return "scene_completeness"
    if any(term in blob for term in ("reasoning_world_knowledge", "reasoning", "world knowledge", "why")):
        return "reasoning_world_knowledge"
    return "other"


def normalize_reference(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, dict):
        return str(value)
    return str(value) if value is not None else ""
