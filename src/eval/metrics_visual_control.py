"""Offline metrics for strict visual-control predictions."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")
CONTROL_SETTINGS = ("text_only", "wrong_image", "blank_image")
def _u(hex_text: str) -> str:
    return bytes.fromhex(hex_text).decode("utf-8")


REFUSAL_PHRASES = (_u("e697a0e6b395e588a4e696ad"), _u("e4b88de883bde7a1aee5ae9a"), _u("e7bcbae5b091e59bbee5838fe4bfa1e681af"), _u("e59bbee5838fe4bfa1e681afe4b88de8b6b3"), "cannot determine", "not enough visual information", "unable to determine")
UNCERTAINTY_PHRASES = REFUSAL_PHRASES + (_u("e4b88de7a1aee5ae9a"), _u("e697a0e6b395e7a1aee5ae9a"), "insufficient visual information", "insufficient information", "cannot tell", "can't determine", "not visible")
HALLUCINATION_PHRASES = (_u("e68891e79c8be588b0"), _u("e59bbee4b8ad"), _u("e794bbe99da2e4b8ad"), "there is", "the image shows", "visible", "on the left", "on the right", "in front", "behind")

def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value.strip())
    except Exception:
        return value


def text_content(value: Any) -> str:
    value = maybe_json(value)
    if isinstance(value, dict):
        return " ".join(str(value.get(k)) for k in ("answer", "reason", "prediction", "refusal") if value.get(k) is not None)
    return str(value or "")


def normalize_text(text: Any) -> str:
    text = text_content(text).lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def tokens(text: Any) -> list[str]:
    norm = normalize_text(text)
    if not norm:
        return []
    if re.search(r"[a-z0-9]", norm):
        return norm.split()
    return list(norm.replace(" ", ""))


def token_f1(prediction: Any, gold: Any) -> float:
    pred_tokens = tokens(prediction)
    gold_tokens = tokens(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: Any, gold: Any) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def answer_length(prediction: Any) -> int:
    text = text_content(prediction)
    return len(text.split()) if re.search(r"[A-Za-z0-9]", text) else len(text.strip())


def is_refusal(text: Any) -> bool:
    lower = text_content(text).lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def has_uncertainty(text: Any) -> bool:
    lower = text_content(text).lower()
    return any(phrase in lower for phrase in UNCERTAINTY_PHRASES)


def is_control_hallucination(text: Any, setting: str) -> bool:
    if setting == "normal":
        return False
    lower = text_content(text).lower()
    return any(phrase in lower for phrase in HALLUCINATION_PHRASES) and not has_uncertainty(lower)


def ensure_scores(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["f1"] = float(out.get("f1")) if out.get("f1") is not None else token_f1(out.get("prediction", ""), out.get("gold", ""))
    out["em"] = float(out.get("em")) if out.get("em") is not None else exact_match(out.get("prediction", ""), out.get("gold", ""))
    out["refusal"] = is_refusal(out.get("prediction", ""))
    out["hallucination"] = is_control_hallucination(out.get("prediction", ""), str(out.get("setting") or ""))
    out["answer_length"] = answer_length(out.get("prediction", ""))
    return out


def group_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        scored = ensure_scores(row)
        sample_id = str(scored.get("id") or "")
        setting = str(scored.get("setting") or "")
        if setting not in SETTINGS:
            raise ValueError(f"unsupported setting for id={sample_id}: {setting}")
        grouped[sample_id][setting] = scored
    missing = {sid: [s for s in SETTINGS if s not in got] for sid, got in grouped.items() if any(s not in got for s in SETTINGS)}
    if missing:
        preview = "; ".join(f"{sid}: {','.join(vals)}" for sid, vals in list(missing.items())[:5])
        raise ValueError(f"missing visual-control settings: {preview}")
    return grouped


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], seed: int = 13, samples: int = 1000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    estimates = [mean([values[rng.randrange(len(values))] for _ in values]) for _ in range(samples)]
    estimates.sort()
    return (estimates[max(0, math.floor(0.025 * len(estimates)))], estimates[min(len(estimates) - 1, math.floor(0.975 * len(estimates)))])


def classify_failure(normal: dict[str, Any], controls: dict[str, dict[str, Any]]) -> str:
    normal_f1 = float(normal["f1"])
    if float(controls["wrong_image"]["f1"]) >= normal_f1:
        return "wrong_image_confound"
    if bool(controls["blank_image"].get("hallucination")):
        return "blank_hallucination"
    if float(controls["text_only"]["f1"]) >= normal_f1:
        return "text_prior_bias"
    if bool(normal.get("refusal")):
        return "over_refusal"
    if normal_f1 > max(float(row["f1"]) for row in controls.values()):
        return "visual_gain"
    return "unresolved"


def case_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by_case(rows)
    cases: list[dict[str, Any]] = []
    for sample_id, settings in grouped.items():
        normal = settings["normal"]
        controls = {key: settings[key] for key in CONTROL_SETTINGS}
        control_scores = {key: float(row["f1"]) for key, row in controls.items()}
        best_control = max(control_scores, key=control_scores.get)
        normal_f1 = float(normal["f1"])
        cases.append({"id": sample_id, "dataset": normal.get("dataset", ""), "model_name": normal.get("model_name", ""), "mode": normal.get("mode", ""), "normal_f1": normal_f1, "text_only_f1": control_scores["text_only"], "wrong_image_f1": control_scores["wrong_image"], "blank_image_f1": control_scores["blank_image"], "case_gap": normal_f1 - control_scores[best_control], "best_control_setting": best_control, "normal_prediction": text_content(normal.get("prediction", "")), "text_only_prediction": text_content(controls["text_only"].get("prediction", "")), "wrong_image_prediction": text_content(controls["wrong_image"].get("prediction", "")), "blank_image_prediction": text_content(controls["blank_image"].get("prediction", "")), "gold": normal.get("gold", ""), "failure_type": classify_failure(normal, controls)})
    return cases


def summarize_rows(rows: list[dict[str, Any]], seed: int = 13) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = [ensure_scores(row) for row in rows]
    cases = case_score_rows(scored)
    by_setting = {setting: [row for row in scored if row.get("setting") == setting] for setting in SETTINGS}
    setting_f1 = {setting: mean([float(row["f1"]) for row in rows_]) for setting, rows_ in by_setting.items()}
    case_gaps = [float(row["case_gap"]) for row in cases]
    control_rows = [row for key in CONTROL_SETTINGS for row in by_setting[key]]
    normal_rows = by_setting["normal"]
    setting_gap_values = [float(c["normal_f1"]) - max(float(c[f"{k}_f1"]) for k in CONTROL_SETTINGS) for c in cases]
    nci = bootstrap_ci([float(row["f1"]) for row in normal_rows], seed=seed)
    sgci = bootstrap_ci(setting_gap_values, seed=seed)
    cgci = bootstrap_ci(case_gaps, seed=seed)
    return ({"count": len(scored), "case_count": len(cases), "normal_f1": setting_f1["normal"], "text_only_f1": setting_f1["text_only"], "wrong_image_f1": setting_f1["wrong_image"], "blank_image_f1": setting_f1["blank_image"], "setting_gap": setting_f1["normal"] - max(setting_f1[key] for key in CONTROL_SETTINGS), "case_gap": mean(case_gaps), "positive_gap_rate": mean([1.0 if gap > 0 else 0.0 for gap in case_gaps]), "normal_refusal_rate": mean([1.0 if row.get("refusal") else 0.0 for row in normal_rows]), "control_hallucination_rate": mean([1.0 if row.get("hallucination") else 0.0 for row in control_rows]), "average_answer_length": mean([float(row.get("answer_length", 0)) for row in scored]), "normal_f1_ci_low": nci[0], "normal_f1_ci_high": nci[1], "setting_gap_ci_low": sgci[0], "setting_gap_ci_high": sgci[1], "case_gap_ci_low": cgci[0], "case_gap_ci_high": cgci[1]}, cases)
