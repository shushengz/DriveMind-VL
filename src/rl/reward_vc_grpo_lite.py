"""Offline visual-control reward harness proposal for future GRPO-lite training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.eval.control_behavior_metrics import behavior_flags, is_refusal_or_caution
from src.eval.metrics_visual_control import answer_length, token_f1
from src.eval.rescore_answer_only import score_prediction

CONTROL_SETTINGS = ("text_only", "wrong_image", "blank_image")


@dataclass(frozen=True)
class RewardWeights:
    lambda_control_f1: float = 0.50
    lambda_blank: float = 0.35
    lambda_text: float = 0.10
    lambda_wrong: float = 0.20
    lambda_refusal: float = 1.00
    lambda_length: float = 0.03


def compute_case_reward(group: dict[str, dict[str, Any]], weights: RewardWeights | None = None) -> dict[str, Any]:
    weights = weights or RewardWeights()
    scored = {setting: score_prediction(group[setting], "answer_only") for setting in ("normal", *CONTROL_SETTINGS)}
    flags = {setting: behavior_flags(scored[setting]["score_text"], scored[setting]["f1"]) for setting in CONTROL_SETTINGS}
    normal_f1 = float(scored["normal"]["f1"])
    control_f1 = {setting: float(scored[setting]["f1"]) for setting in CONTROL_SETTINGS}
    max_control = max(control_f1.values())
    high20 = max_control >= 0.20
    high30 = max_control >= 0.30
    blank_high20 = control_f1["blank_image"] >= 0.20
    blank_high30 = control_f1["blank_image"] >= 0.30
    text_direct = flags["text_only"]["is_direct_answer"] or flags["text_only"]["is_short_prior_answer"]
    blank_direct = flags["blank_image"]["is_direct_answer"] or flags["blank_image"]["is_short_prior_answer"]
    wrong_sim = token_f1(scored["wrong_image"]["score_text"], scored["normal"]["score_text"])
    wrong_high = control_f1["wrong_image"] >= 0.20
    normal_refusal = bool(
        scored["normal"]["refusal"]
        or is_refusal_or_caution(scored["normal"]["score_text"])
    )
    excessive_length = max(0.0, (float(scored["normal"]["answer_length"]) - 25.0) / 25.0)

    normal_reward = normal_f1
    control_penalty = weights.lambda_control_f1 * (max_control + 0.30 * high20 + 0.20 * high30)
    blank_penalty = weights.lambda_blank * (control_f1["blank_image"] + 0.35 * blank_high20 + 0.20 * blank_high30 + 0.10 * blank_direct)
    text_penalty = weights.lambda_text * (0.50 * text_direct + 0.25 * flags["text_only"]["is_short_prior_answer"])
    wrong_penalty = weights.lambda_wrong * (control_f1["wrong_image"] + 0.30 * wrong_high + 0.20 * wrong_sim)
    normal_refusal_penalty = weights.lambda_refusal * float(normal_refusal)
    length_penalty = weights.lambda_length * excessive_length
    total = normal_reward - control_penalty - blank_penalty - text_penalty - wrong_penalty - normal_refusal_penalty - length_penalty
    tags = []
    if normal_f1 >= 0.20:
        tags.append("normal_correct")
    if high20:
        tags.append("control_high_f1")
    if blank_high20:
        tags.append("blank_high_f1")
    if text_direct:
        tags.append("text_direct_answer")
    if wrong_high or wrong_sim >= 0.50:
        tags.append("wrong_image_confound")
    if normal_refusal:
        tags.append("normal_refusal")
    if excessive_length:
        tags.append("long_normal_answer")
    if not tags:
        tags.append("low_signal_case")
    return {
        "id": scored["normal"]["id"],
        "normal_reward": normal_reward,
        "control_penalty": control_penalty,
        "blank_penalty": blank_penalty,
        "text_penalty": text_penalty,
        "wrong_penalty": wrong_penalty,
        "normal_refusal_penalty": normal_refusal_penalty,
        "length_penalty": length_penalty,
        "total_reward": total,
        "reward_tags": tags,
        "normal_f1": normal_f1,
        "text_only_f1": control_f1["text_only"],
        "wrong_image_f1": control_f1["wrong_image"],
        "blank_image_f1": control_f1["blank_image"],
        "max_control_f1": max_control,
        "normal_refusal": normal_refusal,
        "weights": asdict(weights),
    }
