"""Auditable visual-control reward harness v2 for offline GRPO-lite evaluation."""
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
class RewardWeightsV2:
    w_normal: float = 1.0
    w_control: float = 0.8
    w_blank: float = 0.8
    w_text: float = 0.6
    w_wrong: float = 0.8
    w_camera: float = 0.5
    w_spatial: float = 0.7
    w_object: float = 0.6
    w_refusal: float = 0.8
    w_length: float = 0.1


def with_overrides(weights: RewardWeightsV2, **updates: float) -> RewardWeightsV2:
    return replace(weights, **updates)


def _record_parts(record: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if "settings" in record:
        return record["settings"], record
    return record, {
        "id": record.get("normal", {}).get("id", ""),
        "dataset": record.get("normal", {}).get("dataset", "synthetic"),
        "model_name": record.get("normal", {}).get("model_name", "synthetic"),
        "question": record.get("normal", {}).get("question", ""),
        "capability": "",
        "failure_tags": [],
        "image_labels": record.get("normal", {}).get("image_labels", []),
    }


def _tags(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item for item in value.replace(",", "|").split("|") if item}
    return {str(item) for item in (value or []) if str(item)}


def _direction_conflict(gold: str, answer: str) -> bool:
    pairs = (("left", "right"), ("front", "back"), ("ahead", "behind"))
    gold_lower, answer_lower = gold.lower(), answer.lower()
    return any(a in gold_lower and b in answer_lower or b in gold_lower and a in answer_lower for a, b in pairs)


def compute_reward_v2(record: dict[str, Any], weights: RewardWeightsV2 | None = None) -> dict[str, Any]:
    weights = weights or RewardWeightsV2()
    group, meta = _record_parts(record)
    scored = {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}
    flags = {setting: behavior_flags(scored[setting]["score_text"], scored[setting]["f1"]) for setting in CONTROL_SETTINGS}
    normal_f1 = float(scored["normal"]["f1"])
    control = {setting: float(scored[setting]["f1"]) for setting in CONTROL_SETTINGS}
    max_control = max(control.values())
    question = str(meta.get("question") or group["normal"].get("question", ""))
    gold = str(scored["normal"]["gold"] or "")
    dataset = str(meta.get("dataset") or group["normal"].get("dataset", ""))
    failure_tags = _tags(meta.get("failure_tags"))
    drivelm = dataset.lower() == "drivelm"
    spatial_context = drivelm and ("spatial_relation_failure" in failure_tags or meta.get("capability") == "spatial_relation" or bool(SPATIAL_RE.search(question)))
    camera_context = drivelm and ("camera_specific_failure" in failure_tags or bool(CAMERA_RE.search(question)) or bool(meta.get("image_labels")))
    object_context = drivelm and ("object_token_failure" in failure_tags or meta.get("capability") == "object_token" or bool(OBJECT_RE.search(question)))

    normal_raw = normal_f1 + (0.15 if normal_f1 >= 0.30 else 0.0)
    control_raw = max_control
    control_raw += 0.25 if max_control >= 0.20 else 0.0
    control_raw += 0.25 if max_control >= 0.30 else 0.0
    control_raw += 0.25 if max_control >= normal_f1 and max_control >= 0.10 else 0.0

    blank_caution = flags["blank_image"]["is_caution"]
    blank_direct = flags["blank_image"]["is_direct_answer"] or flags["blank_image"]["is_short_prior_answer"]
    blank_raw = 0.0 if blank_caution else control["blank_image"]
    if not blank_caution:
        blank_raw += 0.35 if control["blank_image"] >= 0.20 else 0.0
        blank_raw += 0.20 if control["blank_image"] >= 0.30 else 0.0
        blank_raw += 0.25 if blank_direct else 0.0

    text_caution = flags["text_only"]["is_caution"]
    text_direct = flags["text_only"]["is_direct_answer"] or flags["text_only"]["is_short_prior_answer"]
    text_raw = 0.0 if text_caution else (0.35 if text_direct else 0.0)
    if not text_caution:
        text_raw += 0.35 if control["text_only"] >= 0.20 else 0.0
        text_raw += 0.15 if control["text_only"] >= 0.30 else 0.0

    wrong_similarity = token_f1(scored["wrong_image"]["score_text"], scored["normal"]["score_text"])
    wrong_raw = control["wrong_image"]
    wrong_raw += 0.30 if control["wrong_image"] >= 0.20 else 0.0
    wrong_raw += 0.30 if control["wrong_image"] >= normal_f1 - 0.05 else 0.0
    wrong_raw += 0.20 if wrong_similarity >= 0.50 else 0.0

    wrong_confounded = control["wrong_image"] >= 0.20 or control["wrong_image"] >= normal_f1 - 0.05
    camera_raw = 0.0
    if camera_context and wrong_confounded:
        camera_raw += 0.50
    if camera_context and max_control >= 0.20:
        camera_raw += 0.20

    direction_conflict = _direction_conflict(gold, scored["normal"]["score_text"])
    spatial_raw = 0.0
    if spatial_context:
        spatial_raw += 0.40
        spatial_raw += 0.35 if normal_f1 < 0.20 and max_control >= 0.20 else 0.0
        spatial_raw += 0.30 if direction_conflict else 0.0

    object_raw = 0.0
    if object_context:
        object_raw += 0.40 if "object_token_failure" in failure_tags else 0.20
        object_raw += 0.30 if max(control["wrong_image"], control["text_only"]) >= 0.20 else 0.0

    normal_refusal = bool(scored["normal"]["refusal"] or is_refusal_or_caution(scored["normal"]["score_text"]))
    refusal_raw = float(normal_refusal)
    length_raw = max(0.0, (float(scored["normal"]["answer_length"]) - 25.0) / 25.0)

    components = {
        "normal_reward": weights.w_normal * normal_raw,
        "control_high_f1_penalty": weights.w_control * control_raw,
        "blank_high_f1_penalty": weights.w_blank * blank_raw,
        "text_direct_penalty": weights.w_text * text_raw,
        "wrong_image_penalty": weights.w_wrong * wrong_raw,
        "camera_penalty": weights.w_camera * camera_raw,
        "spatial_penalty": weights.w_spatial * spatial_raw,
        "object_penalty": weights.w_object * object_raw,
        "normal_refusal_penalty": weights.w_refusal * refusal_raw,
        "length_penalty": weights.w_length * length_raw,
    }
    penalties = sum(value for key, value in components.items() if key != "normal_reward")
    total = components["normal_reward"] - penalties
    tags = []
    for enabled, tag in (
        (normal_f1 >= 0.30, "normal_correct_high"),
        (max_control >= 0.20, "control_high_f1"),
        (control["blank_image"] >= 0.20 and not blank_caution, "blank_high_f1"),
        (text_direct and not text_caution, "text_direct_answer"),
        (wrong_confounded, "wrong_image_confound"),
        (camera_raw > 0, "camera_mismatch"),
        (spatial_raw > 0, "spatial_relation_error"),
        (object_raw > 0, "object_token_mismatch"),
        (normal_refusal, "normal_refusal"),
        (length_raw > 0, "long_normal_answer"),
    ):
        if enabled:
            tags.append(tag)
    return {
        "id": str(meta.get("id") or scored["normal"]["id"]),
        "dataset": dataset,
        "model_name": str(meta.get("model_name") or group["normal"].get("model_name", "")),
        "capability": str(meta.get("capability", "")),
        "failure_tags": sorted(failure_tags),
        "normal_f1": normal_f1,
        "text_only_f1": control["text_only"],
        "wrong_image_f1": control["wrong_image"],
        "blank_image_f1": control["blank_image"],
        "case_gap": normal_f1 - max_control,
        **components,
        "total_reward": total,
        "reward_tags": tags,
        "weights": asdict(weights),
    }
