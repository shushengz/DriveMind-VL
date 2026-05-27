"""Reward harness v2.1: calibrated structural penalties and pairwise audit support."""
from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.control_behavior_metrics import behavior_flags, is_refusal_or_caution
from src.eval.metrics_visual_control import token_f1
from src.eval.rescore_answer_only import SETTINGS, score_prediction

CONTROL_SETTINGS = ("text_only", "wrong_image", "blank_image")
SPATIAL_RE = re.compile(r"\b(left|right|front|back|behind|ahead|lane|intersection|relative|position|adjacent)\b", re.I)
CAMERA_RE = re.compile(r"CAM_(?:FRONT|BACK)(?:_(?:LEFT|RIGHT))?|front\s+(?:left|right)\s+camera|back\s+(?:left|right)\s+camera", re.I)
OBJECT_RE = re.compile(r"<c\d+\s*,\s*cam_[^>]+>|object\s+token", re.I)


@dataclass(frozen=True)
class RewardWeightsV21:
    w_normal: float = 1.0
    w_control: float = 0.8
    w_blank: float = 0.65
    w_text: float = 0.6
    w_wrong: float = 0.8
    w_camera: float = 0.7
    w_spatial: float = 0.7
    w_object: float = 0.8
    w_refusal: float = 0.8
    w_length: float = 0.1
    w_caution: float = 0.2


def with_overrides(weights: RewardWeightsV21, **updates: float) -> RewardWeightsV21:
    return replace(weights, **updates)


def _parts(record: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if "settings" in record:
        return record["settings"], record
    normal = record["normal"]
    return record, {
        "id": normal.get("id", ""), "dataset": normal.get("dataset", "synthetic"),
        "model_name": normal.get("model_name", "synthetic"), "question": normal.get("question", ""),
        "capability": "", "failure_tags": [], "image_labels": normal.get("image_labels", []),
    }


def _tag_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item for item in value.replace(",", "|").split("|") if item}
    return {str(item) for item in (value or []) if str(item)}


def _direction_conflict(gold: str, prediction: str) -> bool:
    pairs = (("left", "right"), ("front", "back"), ("ahead", "behind"))
    gold, prediction = gold.lower(), prediction.lower()
    return any((a in gold and b in prediction) or (b in gold and a in prediction) for a, b in pairs)


def compute_reward_v2_1(record: dict[str, Any], weights: RewardWeightsV21 | None = None) -> dict[str, Any]:
    weights = weights or RewardWeightsV21()
    group, meta = _parts(record)
    score = {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}
    flags = {setting: behavior_flags(score[setting]["score_text"], score[setting]["f1"]) for setting in CONTROL_SETTINGS}
    normal = float(score["normal"]["f1"])
    control = {setting: float(score[setting]["f1"]) for setting in CONTROL_SETTINGS}
    max_control = max(control.values())
    tags = _tag_set(meta.get("failure_tags"))
    question = str(meta.get("question") or group["normal"].get("question", ""))
    gold = str(score["normal"]["gold"] or "")
    dataset = str(meta.get("dataset") or group["normal"].get("dataset", ""))
    drivelm = dataset.lower() == "drivelm"

    normal_raw = normal + (0.15 if normal >= 0.30 else 0.0)
    control_raw = max_control + (0.25 if max_control >= 0.20 else 0.0) + (0.25 if max_control >= 0.30 else 0.0)
    control_raw += 0.25 if max_control >= normal and max_control >= 0.10 else 0.0

    blank = control["blank_image"]
    blank_direct = flags["blank_image"]["is_direct_answer"] or flags["blank_image"]["is_short_prior_answer"]
    blank_caution = flags["blank_image"]["is_caution"]
    if blank_caution or blank < 0.20:
        blank_tier, blank_raw = "none", 0.0
    elif blank < 0.30:
        blank_tier, blank_raw = "medium", 0.45
    else:
        blank_tier, blank_raw = "high", 0.70
    if blank_raw and blank_direct:
        blank_raw += 0.20

    text = control["text_only"]
    text_caution = flags["text_only"]["is_caution"]
    text_direct = flags["text_only"]["is_direct_answer"] or flags["text_only"]["is_short_prior_answer"]
    text_raw = 0.0 if text_caution else ((0.35 if text_direct else 0.0) + (0.35 if text >= 0.20 else 0.0) + (0.15 if text >= 0.30 else 0.0))

    wrong = control["wrong_image"]
    wrong_caution = flags["wrong_image"]["is_caution"]
    wrong_similarity = token_f1(score["wrong_image"]["score_text"], score["normal"]["score_text"])
    wrong_confounded = not wrong_caution and (wrong >= 0.20 or wrong >= normal - 0.05 or wrong_similarity >= 0.50)
    wrong_raw = wrong + (0.30 if wrong >= 0.20 else 0.0) + (0.30 if wrong >= normal - 0.05 else 0.0) + (0.20 if wrong_similarity >= 0.50 else 0.0)

    camera_clue = bool(CAMERA_RE.search(question)) or "camera_specific_failure" in tags
    camera_triggered = bool(drivelm and camera_clue and wrong_confounded)
    camera_raw = (0.45 + (0.25 if wrong >= 0.20 else 0.0)) if camera_triggered else 0.0

    object_clue = bool(OBJECT_RE.search(question)) or "object_token_failure" in tags or meta.get("capability") == "object_token"
    object_control = any(control[setting] >= 0.20 and not flags[setting]["is_caution"] for setting in CONTROL_SETTINGS)
    object_triggered = bool(drivelm and object_clue and (object_control or wrong_confounded))
    object_raw = (0.40 + (0.30 if object_control else 0.0) + (0.15 if wrong_confounded else 0.0)) if object_triggered else 0.0

    spatial_clue = bool(SPATIAL_RE.search(question)) or "spatial_relation_failure" in tags or meta.get("capability") == "spatial_relation"
    direction_conflict = _direction_conflict(gold, score["normal"]["score_text"])
    severe_spatial = spatial_clue and ((normal < 0.20 and max_control >= 0.20) or (wrong >= 0.20 and bool(SPATIAL_RE.search(question))) or direction_conflict)
    spatial_triggered = bool(drivelm and severe_spatial)
    spatial_raw = (0.55 + (0.25 if direction_conflict else 0.0) + (0.20 if wrong >= 0.20 else 0.0)) if spatial_triggered else 0.0

    normal_refusal = bool(score["normal"]["refusal"] or is_refusal_or_caution(score["normal"]["score_text"]))
    valid_cautions = sum(flags[setting]["is_caution"] for setting in CONTROL_SETTINGS)
    caution_raw = (valid_cautions / 3.0) if normal >= 0.20 and not normal_refusal else 0.0
    length_raw = max(0.0, (float(score["normal"]["answer_length"]) - 25.0) / 25.0)
    components = {
        "normal_reward": weights.w_normal * normal_raw,
        "control_high_f1_penalty": weights.w_control * control_raw,
        "blank_high_f1_penalty": weights.w_blank * blank_raw,
        "text_direct_penalty": weights.w_text * text_raw,
        "wrong_image_penalty": weights.w_wrong * wrong_raw,
        "camera_penalty": weights.w_camera * camera_raw,
        "spatial_penalty": weights.w_spatial * spatial_raw,
        "object_penalty": weights.w_object * object_raw,
        "normal_refusal_penalty": weights.w_refusal * float(normal_refusal),
        "length_penalty": weights.w_length * length_raw,
        "valid_control_caution_reward": weights.w_caution * caution_raw,
    }
    total = components["normal_reward"] + components["valid_control_caution_reward"] - sum(
        value for key, value in components.items() if key.endswith("_penalty")
    )
    reward_tags = []
    for yes, tag in (
        (normal >= 0.30, "normal_correct_high"), (max_control >= 0.20, "control_high_f1"),
        (blank_raw > 0, "blank_high_f1"), (text_direct and not text_caution, "text_direct_answer"),
        (wrong_confounded, "wrong_image_confound"), (camera_triggered, "camera_grounding_mismatch"),
        (spatial_triggered, "spatial_relation_error"), (object_triggered, "object_grounding_mismatch"),
        (normal_refusal, "normal_refusal"), (caution_raw > 0, "valid_control_caution"),
        (length_raw > 0, "long_normal_answer"),
    ):
        if yes:
            reward_tags.append(tag)
    return {
        "id": str(meta.get("id") or score["normal"]["id"]), "dataset": dataset,
        "model_name": str(meta.get("model_name") or group["normal"].get("model_name", "")),
        "capability": str(meta.get("capability", "")), "failure_tags": sorted(tags),
        "normal_f1": normal, "text_only_f1": text, "wrong_image_f1": wrong, "blank_image_f1": blank,
        "case_gap": normal - max_control, **components, "total_reward": total, "reward_tags": reward_tags,
        "blank_penalty_tier": blank_tier, "camera_grounding_triggered": camera_triggered,
        "object_grounding_triggered": object_triggered, "spatial_grounding_triggered": spatial_triggered,
        "pairwise_ready": True, "weights": asdict(weights),
    }


def compute_pairwise_preference_reward(record_a: dict[str, Any], record_b: dict[str, Any], weights: RewardWeightsV21 | None = None) -> dict[str, Any]:
    reward_a = compute_reward_v2_1(record_a, weights)["total_reward"]
    reward_b = compute_reward_v2_1(record_b, weights)["total_reward"]
    margin = reward_a - reward_b
    preferred = "a" if margin > 1e-9 else ("b" if margin < -1e-9 else "tie")
    return {"preferred": preferred, "reward_a": reward_a, "reward_b": reward_b, "margin": margin}
