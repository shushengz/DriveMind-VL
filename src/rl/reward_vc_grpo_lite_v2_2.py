"""Reward harness v2.2: integrate human-reviewed structural failure patches."""
from __future__ import annotations

import re
import string
from dataclasses import asdict, dataclass, replace
from typing import Any

from src.eval.control_behavior_metrics import is_refusal_or_caution
from src.eval.rescore_answer_only import SETTINGS, score_prediction
from src.rl.reward_vc_grpo_lite_v2_1 import RewardWeightsV21, compute_reward_v2_1

CONTROL_SETTINGS = ("text_only", "wrong_image", "blank_image")
OBJECT_TOKEN_RE = re.compile(r"<c[^>]*>|object\s*token", re.I)
INVALID_PATTERNS = {
    "terminate": re.compile(r"\bterminate(?:d)?\b", re.I),
    "no further actions needed": re.compile(r"\bno further actions? (?:are )?needed\b", re.I),
    "task completed": re.compile(r"\btask completed\b", re.I),
    "mission completed": re.compile(r"\bmission completed\b", re.I),
    "validation set": re.compile(r"\bvalidation set\b", re.I),
    "training set": re.compile(r"\btraining set\b", re.I),
    "model output": re.compile(r"\bmodel output\b", re.I),
    "camera resolution": re.compile(r"\bcamera resolution\b", re.I),
    "front camera only": re.compile(r"\bfront camera only\b", re.I),
    "bare front camera": re.compile(r"^\s*(?:the\s+)?front camera[.]?\s*$", re.I),
    "unable due to system": re.compile(r"\bunable due to (?:the )?system\b", re.I),
    "placeholder": re.compile(r"\bplaceholder\b", re.I),
    "null": re.compile(r"^\s*null\s*$", re.I),
    "n/a": re.compile(r"^\s*n/?a\s*$", re.I),
}
CATEGORY_TERMS = {
    "pedestrian": ("pedestrian", "person", "people", "man", "woman", "walker"),
    "cyclist": ("cyclist", "bicycle", "bike", "rider"),
    "car": ("car", "sedan", "suv"),
    "vehicle": ("vehicle",),
    "truck": ("truck", "lorry"),
    "bus": ("bus",),
    "motorcycle": ("motorcycle", "motorbike", "scooter"),
    "traffic_light": ("traffic light", "red light", "green light", "yellow light"),
    "traffic_sign": ("traffic sign", "stop sign", "sign"),
    "barrier": ("road barrier", "barrier", "cone", "bollard"),
    "lane": ("left lane", "right lane", "current lane", "lane"),
}
VEHICLE_FAMILY = {"car", "vehicle", "truck", "bus", "motorcycle"}
ACTION_QUESTION_RE = re.compile(r"\b(action|actions|collision)\b", re.I)
GENERIC_ACTION_RE = re.compile(r"\b(slow down|stop|wait|brake)\b", re.I)


@dataclass(frozen=True)
class RewardWeightsV22:
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
    w_obj_inv: float = 0.7
    w_obj_cat: float = 0.5
    w_invalid: float = 0.8
    w_same: float = 0.6


def with_overrides(weights: RewardWeightsV22, **updates: float) -> RewardWeightsV22:
    return replace(weights, **updates)


def _base_weights(weights: RewardWeightsV22) -> RewardWeightsV21:
    return RewardWeightsV21(**{field: getattr(weights, field) for field in RewardWeightsV21.__dataclass_fields__})


def _parts(record: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if "settings" in record:
        return record["settings"], record
    normal = record["normal"]
    return record, {
        "id": normal.get("id", ""),
        "dataset": normal.get("dataset", "synthetic"),
        "model_name": normal.get("model_name", "synthetic"),
        "question": normal.get("question", ""),
        "failure_tags": [],
        "capability": "",
    }


def _tag_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {tag for tag in value.replace(",", "|").split("|") if tag}
    return {str(tag) for tag in (value or []) if str(tag)}


def normalize_answer(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r'^\s*\{\s*"answer"\s*:\s*"(.*?)"\s*\}\s*$', r"\1", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def answer_similarity(a: str, b: str) -> float:
    left, right = set(normalize_answer(a).split()), set(normalize_answer(b).split())
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _categories(text: str) -> set[str]:
    cleaned = normalize_answer(text)
    categories = set()
    for category, terms in CATEGORY_TERMS.items():
        if any(re.search(rf"\b{re.escape(term)}\b", cleaned) for term in terms):
            categories.add(category)
    return categories


def _category_conflict(gold: set[str], answer: set[str]) -> str:
    if not gold or not answer:
        return ""
    specific_gold = gold - {"vehicle", "lane"}
    specific_answer = answer - {"vehicle", "lane"}
    if "vehicle" in gold and answer & VEHICLE_FAMILY:
        return ""
    if "vehicle" in answer and gold & VEHICLE_FAMILY:
        return ""
    if specific_gold and specific_answer and not (specific_gold & specific_answer):
        return f"gold={','.join(sorted(specific_gold))}; answer={','.join(sorted(specific_answer))}"
    return ""


def _invalid_pattern(answer: str) -> str:
    if is_refusal_or_caution(answer):
        return ""
    for label, pattern in INVALID_PATTERNS.items():
        if pattern.search(str(answer or "")):
            return label
    return ""


def compute_reward_v2_2(record: dict[str, Any], weights: RewardWeightsV22 | None = None) -> dict[str, Any]:
    weights = weights or RewardWeightsV22()
    group, meta = _parts(record)
    base = compute_reward_v2_1(record, _base_weights(weights))
    scores = {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}
    answers = {setting: str(scores[setting]["score_text"] or "") for setting in SETTINGS}
    control_f1 = {setting: float(scores[setting]["f1"]) for setting in CONTROL_SETTINGS}
    tags = _tag_set(meta.get("failure_tags"))
    dataset = str(base["dataset"]).lower()
    question = str(meta.get("question") or group["normal"].get("question", ""))
    object_clue = bool(OBJECT_TOKEN_RE.search(question)) or "object_token_failure" in tags or meta.get("capability") == "object_token"

    similarities = {setting: answer_similarity(answers["normal"], answers[setting]) for setting in CONTROL_SETTINGS}
    usable_similar = [
        setting for setting in CONTROL_SETTINGS
        if similarities[setting] >= 0.75 and not is_refusal_or_caution(answers[setting])
    ]
    object_invariant_triggered = bool(
        dataset == "drivelm"
        and object_clue
        and not is_refusal_or_caution(answers["normal"])
        and len(usable_similar) >= 2
    )
    generic_action_answers = [
        setting for setting in SETTINGS
        if GENERIC_ACTION_RE.search(answers[setting]) and not is_refusal_or_caution(answers[setting])
    ]
    semantic_action_invariant = bool(
        dataset == "drivelm"
        and object_clue
        and ACTION_QUESTION_RE.search(question)
        and len(generic_action_answers) >= 3
    )
    object_invariant_triggered = object_invariant_triggered or semantic_action_invariant
    obj_inv_raw = 0.0
    if object_invariant_triggered:
        obj_inv_raw = 0.65 + (0.25 if max(control_f1.values()) >= 0.20 else 0.0) + (0.10 if len(usable_similar) == 3 else 0.0)

    gold_categories = _categories(str(scores["normal"]["gold"] or ""))
    answer_categories = _categories(answers["normal"])
    conflict_reason = "" if is_refusal_or_caution(answers["normal"]) else _category_conflict(gold_categories, answer_categories)
    category_mismatch_triggered = bool(conflict_reason)
    obj_cat_raw = 1.0 if category_mismatch_triggered else 0.0

    invalid_hits: dict[str, str] = {}
    for setting in SETTINGS:
        pattern = _invalid_pattern(answers[setting])
        if pattern:
            invalid_hits[setting] = pattern
    invalid_raw = min(2.0, (1.0 if "normal" in invalid_hits else 0.0) + 0.60 * sum(setting != "normal" for setting in invalid_hits))

    same_settings = [
        setting for setting in CONTROL_SETTINGS
        if similarities[setting] >= 0.80 and not is_refusal_or_caution(answers[setting])
    ]
    same_raw = min(
        1.50,
        sum(0.45 + (0.25 if control_f1[setting] >= 0.20 else 0.0) for setting in same_settings),
    )

    patches = {
        "object_token_invariant_penalty": weights.w_obj_inv * obj_inv_raw,
        "normal_object_category_mismatch_penalty": weights.w_obj_cat * obj_cat_raw,
        "invalid_generic_answer_penalty": weights.w_invalid * invalid_raw,
        "control_same_as_normal_penalty": weights.w_same * same_raw,
    }
    result = dict(base)
    result.update(patches)
    result["total_reward"] = base["total_reward"] - sum(patches.values())
    result["reward_v2_1_total_reward"] = base["total_reward"]
    result["reward_v2_2_delta"] = result["total_reward"] - base["total_reward"]
    result["object_token_invariant_triggered"] = object_invariant_triggered
    result["object_token_semantic_action_invariant_triggered"] = semantic_action_invariant
    result["object_token_answer_similarity_max"] = max(similarities.values())
    result["object_token_answer_similarity_mean"] = sum(similarities.values()) / len(similarities)
    result["normal_object_category_mismatch_triggered"] = category_mismatch_triggered
    result["gold_object_categories"] = sorted(gold_categories)
    result["answer_object_categories"] = sorted(answer_categories)
    result["object_category_conflict_reason"] = conflict_reason
    result["invalid_generic_answer_triggered"] = bool(invalid_hits)
    result["invalid_generic_answer_setting"] = sorted(invalid_hits)
    result["invalid_generic_pattern"] = invalid_hits
    result["control_same_as_normal_triggered"] = bool(same_settings)
    result["same_as_normal_settings"] = same_settings
    result["same_as_normal_similarity"] = similarities
    result["weights"] = asdict(weights)
    new_tags = list(result["reward_tags"])
    for active, tag in (
        (object_invariant_triggered, "object_token_invariant_answer"),
        (category_mismatch_triggered, "normal_object_category_mismatch"),
        (bool(invalid_hits), "invalid_generic_answer"),
        (bool(same_settings), "control_same_as_normal"),
    ):
        if active and tag not in new_tags:
            new_tags.append(tag)
    result["reward_tags"] = new_tags
    return result


def compute_pairwise_preference_reward(record_a: dict[str, Any], record_b: dict[str, Any], weights: RewardWeightsV22 | None = None) -> dict[str, Any]:
    reward_a = compute_reward_v2_2(record_a, weights)["total_reward"]
    reward_b = compute_reward_v2_2(record_b, weights)["total_reward"]
    margin = reward_a - reward_b
    return {"preferred": "a" if margin > 1e-9 else ("b" if margin < -1e-9 else "tie"), "reward_a": reward_a, "reward_b": reward_b, "margin": margin}
