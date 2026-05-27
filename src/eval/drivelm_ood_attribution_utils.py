"""Shared offline helpers for Stage 13.5 DriveLM OOD failure attribution."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.build_drivelm_ood_eval_pool import classify_question
from src.eval.control_behavior_metrics import behavior_flags, is_refusal_or_caution
from src.eval.rescore_answer_only import SETTINGS, align_settings, load_model_settings, score_prediction
from src.eval.metrics_visual_control import token_f1

BASE = "base_qwen25vl_3b"
R3 = "sft_v3_r3_lingo_smoke"
CONTROL_SETTINGS = ("text_only", "wrong_image", "blank_image")
PREDICTION_ROOT = Path("outputs/predictions_drivelm_ood")
SPATIAL_RE = re.compile(r"\b(left|right|front|back|behind|ahead|lane|intersection|relative|position|adjacent|side)\b", re.I)
LEFT_RIGHT_RE = re.compile(r"\b(left|right)\b", re.I)
FRONT_BACK_RE = re.compile(r"\b(front|back|behind|ahead)\b", re.I)
LANE_RE = re.compile(r"\b(lane|adjacent lane|current lane)\b", re.I)
OBJECT_RE = re.compile(r"<c\d+\s*,\s*cam_[^>]+>", re.I)
CAMERA_RE = re.compile(r"CAM_(?:FRONT|BACK)(?:_(?:LEFT|RIGHT))?", re.I)


def groups(model: str, prediction_root: Path = PREDICTION_ROOT) -> dict[str, dict[str, dict[str, Any]]]:
    aligned = align_settings(load_model_settings(prediction_root, "drivelm", model, "strict_visual"))
    return {str(group["normal"]["id"]): group for group in aligned}


def scored(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}


def case_gap(scores: dict[str, dict[str, Any]]) -> float:
    return float(scores["normal"]["f1"]) - max(float(scores[setting]["f1"]) for setting in CONTROL_SETTINGS)


def high_control(scores: dict[str, dict[str, Any]], threshold: float = 0.20) -> bool:
    return any(float(scores[setting]["f1"]) >= threshold for setting in CONTROL_SETTINGS)


def answer(scores: dict[str, dict[str, Any]], setting: str) -> str:
    return str(scores[setting]["score_text"] or "")


def direct_prior(scores: dict[str, dict[str, Any]], setting: str) -> bool:
    flags = behavior_flags(answer(scores, setting), float(scores[setting]["f1"]))
    return bool((flags["is_direct_answer"] or flags["is_short_prior_answer"]) and not flags["is_caution"])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["message"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def load_cases(prediction_root: Path = PREDICTION_ROOT) -> list[dict[str, Any]]:
    base_groups, r3_groups = groups(BASE, prediction_root), groups(R3, prediction_root)
    common = sorted(set(base_groups) & set(r3_groups))
    cases: list[dict[str, Any]] = []
    for sample_id in common:
        raw = r3_groups[sample_id]
        base, r3 = scored(base_groups[sample_id]), scored(raw)
        question = str(raw["normal"].get("question", ""))
        labels = raw["normal"].get("image_labels", [])
        cases.append({
            "id": sample_id,
            "capability": classify_question(question),
            "question": question,
            "gold": str(raw["normal"].get("gold", "")),
            "image_labels": labels,
            "raw": raw,
            "base": base,
            "r3": r3,
            "base_case_gap": case_gap(base),
            "r3_case_gap": case_gap(r3),
            "delta_normal_f1": float(r3["normal"]["f1"]) - float(base["normal"]["f1"]),
            "delta_case_gap": case_gap(r3) - case_gap(base),
            "is_spatial_question": bool(SPATIAL_RE.search(question)),
            "is_camera_question": bool(CAMERA_RE.search(question)),
            "has_camera_label": bool(labels),
            "has_object_token": bool(OBJECT_RE.search(question)),
        })
    return cases


def tag_case(case: dict[str, Any]) -> list[str]:
    r3 = case["r3"]
    base = case["base"]
    question = case["question"]
    tags: list[str] = []
    normal = float(r3["normal"]["f1"])
    text = float(r3["text_only"]["f1"])
    wrong = float(r3["wrong_image"]["f1"])
    blank = float(r3["blank_image"]["f1"])
    wrong_similarity = token_f1(answer(r3, "wrong_image"), answer(r3, "normal"))
    control_high = high_control(r3)

    if case["is_spatial_question"] and (normal < 0.20 or control_high or wrong >= normal - 0.05):
        tags.append("spatial_relation_failure")
    if (case["is_camera_question"] or case["has_camera_label"]) and (wrong >= 0.20 or wrong >= normal - 0.05):
        tags.append("camera_specific_failure")
    if case["has_object_token"] and (text >= 0.20 or wrong >= 0.20 or wrong >= normal - 0.05):
        tags.append("object_token_failure")
    if blank >= 0.20 and direct_prior(r3, "blank_image"):
        tags.append("blank_prior_answer")
    if text >= 0.20 and direct_prior(r3, "text_only"):
        tags.append("text_only_prior_answer")
    if wrong >= 0.20 or wrong >= normal - 0.05 or wrong_similarity >= 0.50:
        tags.append("wrong_image_confound")
    if case["delta_normal_f1"] > 0.10 and case["delta_case_gap"] <= 0:
        tags.append("normal_answer_style_transfer")
    if float(base["normal"]["f1"]) >= 0.30 and case["delta_normal_f1"] < -0.10:
        tags.append("normal_regression")
    if normal > max(text, wrong, blank) and case["delta_normal_f1"] > 0:
        tags.append("visual_gain_success")
    return tags or ["review_other"]


PRIMARY_ORDER = [
    "normal_regression", "blank_prior_answer", "wrong_image_confound",
    "text_only_prior_answer", "spatial_relation_failure",
    "camera_specific_failure", "object_token_failure",
    "normal_answer_style_transfer", "visual_gain_success", "review_other",
]


def primary_type(tags: list[str]) -> str:
    return next((tag for tag in PRIMARY_ORDER if tag in tags), tags[0])


COMMENTS = {
    "spatial_relation_failure": "空间关系问题的 normal 证据弱或控制条件同样高分，说明位置关系尚未可靠落到图像上。",
    "camera_specific_failure": "错误视角下仍可得到高重合答案，camera label 与视觉视角对齐存在风险。",
    "object_token_failure": "对象引用问题在无图或错图下仍高重合，object token grounding 尚不可靠。",
    "blank_prior_answer": "空白图下仍直接回答并与 gold 高重合，是明确的语言先验失败。",
    "text_only_prior_answer": "无图输入仍直接作答且高重合，说明答案可被语言模板触发。",
    "wrong_image_confound": "错图输出与 gold 或 normal 回答高度一致，模型没有稳定使用正确视觉证据。",
    "normal_answer_style_transfer": "r3 提升了正常回答分数，但未改善 case gap，属于回答风格迁移而非 grounding 提升。",
    "normal_regression": "Base 的正常回答较好而 r3 退化，属于跨域正常能力损失。",
    "visual_gain_success": "r3 的 normal 优于控制输入且优于 Base，是少量可信视觉增益案例。",
    "review_other": "该样本没有触发主要规则，保留供人工复核。",
}


def taxonomy_row(case: dict[str, Any]) -> dict[str, Any]:
    tags = tag_case(case)
    r3, base = case["r3"], case["base"]
    primary = primary_type(tags)
    return {
        "id": case["id"],
        "capability": case["capability"],
        "question": case["question"],
        "gold": case["gold"],
        "image_labels": json.dumps(case["image_labels"], ensure_ascii=False),
        "base_normal_answer": answer(base, "normal"),
        "r3_normal_answer": answer(r3, "normal"),
        "r3_text_answer": answer(r3, "text_only"),
        "r3_wrong_answer": answer(r3, "wrong_image"),
        "r3_blank_answer": answer(r3, "blank_image"),
        "base_normal_f1": float(base["normal"]["f1"]),
        "r3_normal_f1": float(r3["normal"]["f1"]),
        "r3_text_f1": float(r3["text_only"]["f1"]),
        "r3_wrong_f1": float(r3["wrong_image"]["f1"]),
        "r3_blank_f1": float(r3["blank_image"]["f1"]),
        "r3_case_gap": case["r3_case_gap"],
        "base_case_gap": case["base_case_gap"],
        "delta_normal_f1": case["delta_normal_f1"],
        "delta_case_gap": case["delta_case_gap"],
        "failure_tags": "|".join(tags),
        "primary_failure_type": primary,
        "chinese_comment": COMMENTS[primary],
    }
