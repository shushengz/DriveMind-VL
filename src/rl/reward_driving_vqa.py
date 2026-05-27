"""Offline reward components for driving VQA visual-control predictions."""

from __future__ import annotations

import json
from typing import Any

from src.eval.metrics_visual_control import CONTROL_SETTINGS, answer_length, is_control_hallucination, is_refusal, token_f1


def parse_answer_json(prediction: Any) -> tuple[dict[str, str], bool]:
    if isinstance(prediction, dict):
        return {"answer": str(prediction.get("answer", "")), "reason": str(prediction.get("reason", ""))}, False
    if isinstance(prediction, str):
        try:
            obj = json.loads(prediction)
            if isinstance(obj, dict) and "answer" in obj and "reason" in obj:
                return {"answer": str(obj.get("answer", "")), "reason": str(obj.get("reason", ""))}, False
        except Exception:
            pass
        return {"answer": prediction, "reason": ""}, True
    return {"answer": str(prediction or ""), "reason": ""}, True


def compute_reward(row: dict[str, Any], case_group: dict[str, dict[str, Any]] | None = None, weights: dict[str, float] | None = None) -> dict[str, Any]:
    weights = weights or {"lambda_gap": 0.5, "lambda_control": 0.4, "lambda_format": 0.2, "lambda_refusal": 0.6, "lambda_length": 0.02}
    setting = str(row.get("setting") or "normal")
    pred_obj, format_error = parse_answer_json(row.get("prediction", ""))
    pred_text = " ".join([pred_obj.get("answer", ""), pred_obj.get("reason", "")]).strip()
    f1 = token_f1(pred_obj.get("answer", pred_text), row.get("gold", ""))
    r_answer = f1 if setting == "normal" else 0.0
    r_visual_gap = 0.0
    if case_group and "normal" in case_group:
        normal_pred, _ = parse_answer_json(case_group["normal"].get("prediction", ""))
        normal_f1 = token_f1(normal_pred.get("answer", ""), case_group["normal"].get("gold", row.get("gold", "")))
        control_f1s = []
        for key in CONTROL_SETTINGS:
            if key in case_group:
                c_pred, _ = parse_answer_json(case_group[key].get("prediction", ""))
                control_f1s.append(token_f1(c_pred.get("answer", ""), case_group[key].get("gold", row.get("gold", ""))))
        if control_f1s:
            gap = normal_f1 - max(control_f1s)
            r_visual_gap = gap if setting == "normal" else min(0.0, gap)
    hallucination = is_control_hallucination(pred_text, setting)
    refusal = is_refusal(pred_text)
    r_control = (-1.0 if hallucination else (0.5 if refusal else 0.0)) if setting in CONTROL_SETTINGS else 0.0
    r_format = 1.0 if not format_error and pred_obj.get("answer") and pred_obj.get("reason") else (0.3 if pred_obj.get("answer") else 0.0)
    r_over_refusal = 1.0 if setting == "normal" and refusal else 0.0
    length = answer_length(pred_text)
    r_length = max(0.0, (length - 80) / 80)
    total = r_answer + weights["lambda_gap"] * r_visual_gap + weights["lambda_control"] * r_control + weights["lambda_format"] * r_format - weights["lambda_refusal"] * r_over_refusal - weights["lambda_length"] * r_length
    return {"id": row.get("id", ""), "setting": setting, "reward": total, "r_answer": r_answer, "r_visual_gap": r_visual_gap, "r_control_calibration": r_control, "r_format": r_format, "r_over_refusal": r_over_refusal, "r_length": r_length, "f1": f1, "hallucination": hallucination, "refusal": refusal, "format_error": format_error, "answer_length": length}


def reward_hacking_warnings(rewards: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in rewards:
        by_setting.setdefault(str(row.get("setting")), []).append(row)
    normal = by_setting.get("normal", [])
    controls = [row for key in CONTROL_SETTINGS for row in by_setting.get(key, [])]
    if normal:
        normal_f1 = sum(float(r.get("f1", 0.0)) for r in normal) / len(normal)
        normal_refusal = sum(1 for r in normal if r.get("refusal")) / len(normal)
        if normal_f1 < 0.2 and controls and sum(float(r.get("reward", 0.0)) for r in controls) / len(controls) > 0.4:
            warnings.append("normal F1 low but control reward high")
        if normal_refusal > 0.3:
            warnings.append("normal refusal rate too high")
        if sum(float(r.get("answer_length", 0.0)) for r in normal) / len(normal) > 120:
            warnings.append("answer length unusually long")
    return warnings
