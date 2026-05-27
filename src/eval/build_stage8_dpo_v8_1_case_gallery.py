"""Build a held-out case gallery comparing r3, DPO-v8 and DPO-v8.1."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rescore_answer_only import SETTINGS, align_settings, load_model_settings, score_prediction

R3, DPO8 = "sft_v3_r3_lingo_smoke", "dpo_v8_lingo_smoke"
STEP25, STEP50 = "dpo_v8_1_lingo_smoke_step25", "dpo_v8_1_lingo_smoke_step50"
SPATIAL = re.compile(r"left|right|front|back|lane|traffic light|pedestrian|vehicle|behind|ahead", re.I)


def model_cases(root: Path, model: str) -> dict[str, dict[str, dict[str, Any]]]:
    aligned = align_settings(load_model_settings(root, "lingoqa", model, "strict_visual"))
    return {str(group["normal"]["id"]): group for group in aligned}


def scores(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}


def gap(value: dict[str, dict[str, Any]]) -> float:
    return value["normal"]["f1"] - max(value[s]["f1"] for s in ("text_only", "wrong_image", "blank_image"))


def category(group: dict[str, dict[str, Any]], r3: dict[str, dict[str, Any]], dpo8: dict[str, dict[str, Any]], best: dict[str, dict[str, Any]]) -> str:
    if any(r3[s]["hallucination"] and not best[s]["hallucination"] for s in ("text_only", "wrong_image", "blank_image")):
        return "r3_hallucination_v8_1_fixed"
    if any(dpo8[s]["hallucination"] and not best[s]["hallucination"] for s in ("text_only", "wrong_image", "blank_image")):
        return "dpo_v8_unfixed_v8_1_fixed"
    if r3["normal"]["f1"] > best["normal"]["f1"] + 0.2:
        return "r3_correct_v8_1_wrong"
    if best["normal"]["f1"] >= 0.7 and best["wrong_image"]["f1"] >= best["normal"]["f1"] - 0.05:
        return "v8_1_normal_ok_wrong_ok"
    if best["blank_image"]["hallucination"]:
        return "v8_1_blank_hallucination"
    if best["text_only"]["f1"] >= best["normal"]["f1"]:
        return "v8_1_text_prior_bias"
    if best["normal"]["refusal"]:
        return "v8_1_over_refusal"
    if gap(best) > gap(r3) + 0.2:
        return "v8_1_case_gap_improved"
    if SPATIAL.search(str(group["normal"].get("question", ""))) and best["normal"]["f1"] < 0.5:
        return "v8_1_spatial_failure"
    return "unresolved"


COMMENTS = {
    "r3_hallucination_v8_1_fixed": "r3 在控制条件下出现强断言，而 model-mined DPO-v8.1 修正了该行为，说明真实错误偏好对该样本有效。",
    "dpo_v8_unfixed_v8_1_fixed": "rule-based DPO-v8 未消除该控制组问题，v8.1 使用真实 rejected 后完成修正。",
    "r3_correct_v8_1_wrong": "r3 在 normal 输入下更好，而 v8.1 退化，需警惕 control pairs 过强损伤正常问答。",
    "v8_1_normal_ok_wrong_ok": "v8.1 在 normal 下正确，但 wrong-image 下仍接近答案，仍存在错误证据混淆风险。",
    "v8_1_blank_hallucination": "v8.1 在 blank-image 下仍给出强断言，blank prior 尚未完全消除。",
    "v8_1_text_prior_bias": "v8.1 在 text-only 下仍与 normal 竞争，说明语言先验仍需进一步约束。",
    "v8_1_over_refusal": "v8.1 在 normal 下拒答，说明偏好校准出现过度谨慎风险。",
    "v8_1_case_gap_improved": "v8.1 拉开了 normal 与控制组表现差距，是 visual dependency 的正向样本。",
    "v8_1_spatial_failure": "该空间关系问题仍失败，后续需要保留或加强 spatial anchor。",
    "unresolved": "该样本保留用于人工复核 raw predictions，尚不能归入主要失败类型。",
}


def output_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["message"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def output_html(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["message"]
    parts = ["<html><head><meta charset='utf-8'><style>table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;max-width:360px}</style></head><body><h1>Stage 8 DPO-v8.1 Held-out Case Gallery</h1><table><tr>"]
    parts.extend(f"<th>{html.escape(field)}</th>" for field in fields)
    parts.append("</tr>")
    for row in rows:
        parts.append("<tr>")
        parts.extend(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        parts.append("</tr>")
    parts.append("</table></body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Stage 8 DPO-v8.1 case gallery.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--diagnosis", default="outputs/final_report/stage8_dpo_v8_1_diagnosis.json")
    parser.add_argument("--output_csv", default="outputs/final_report/stage8_dpo_v8_1_case_gallery.csv")
    parser.add_argument("--output_html", default="outputs/final_report/stage8_dpo_v8_1_case_gallery.html")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    root = Path(args.prediction_root)
    try:
        diagnosis = json.loads(Path(args.diagnosis).read_text(encoding="utf-8"))
        best_name = diagnosis.get("best_checkpoint") or STEP25
        groups = {model: model_cases(root, model) for model in (R3, DPO8, best_name)}
    except Exception as exc:
        rows = [{"message": f"DPO-v8.1 held-out predictions are not available: {exc}"}]
        output_rows(Path(args.output_csv), rows)
        output_html(Path(args.output_html), rows)
        print(json.dumps({"count": 0, "message": rows[0]["message"]}, ensure_ascii=False, indent=2))
        return
    cases = []
    for sample_id in sorted(set(groups[R3]) & set(groups[DPO8]) & set(groups[best_name])):
        r3, dpo8, best = (scores(groups[name][sample_id]) for name in (R3, DPO8, best_name))
        kind = category(groups[best_name][sample_id], r3, dpo8, best)
        cases.append({
            "id": sample_id, "best_checkpoint": best_name, "case_type": kind,
            "question": groups[best_name][sample_id]["normal"].get("question", ""),
            "gold": groups[best_name][sample_id]["normal"].get("gold", ""),
            "r3_normal_f1": r3["normal"]["f1"], "dpo_v8_normal_f1": dpo8["normal"]["f1"], "v8_1_normal_f1": best["normal"]["f1"],
            "r3_case_gap": gap(r3), "v8_1_case_gap": gap(best),
            "normal_prediction": groups[best_name][sample_id]["normal"].get("prediction", ""),
            "text_only_prediction": groups[best_name][sample_id]["text_only"].get("prediction", ""),
            "wrong_image_prediction": groups[best_name][sample_id]["wrong_image"].get("prediction", ""),
            "blank_image_prediction": groups[best_name][sample_id]["blank_image"].get("prediction", ""),
            "chinese_comment": COMMENTS[kind],
        })
    cases.sort(key=lambda row: (row["case_type"] == "unresolved", row["v8_1_case_gap"] - row["r3_case_gap"]))
    cases = cases[: args.limit]
    output_rows(Path(args.output_csv), cases)
    output_html(Path(args.output_html), cases)
    print(json.dumps({"best_checkpoint": best_name, "count": len(cases), "csv": args.output_csv, "html": args.output_html}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
