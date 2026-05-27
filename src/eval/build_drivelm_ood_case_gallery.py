"""Build an HTML/CSV gallery for Stage 13 DriveLM OOD diagnosis."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.build_drivelm_ood_eval_pool import classify_question
from src.eval.rescore_answer_only import SETTINGS, align_settings, load_model_settings, score_prediction

BASE = "base_qwen25vl_3b"
R3 = "sft_v3_r3_lingo_smoke"
FAILURE_TYPES = [
    "base_wrong_r3_correct", "base_correct_r3_wrong",
    "r3_normal_text_prior", "r3_normal_wrong_image_confound", "r3_blank_high_f1",
    "camera_specific_failure", "object_token_failure", "spatial_relation_failure",
    "counting_failure", "action_reasoning_failure",
]
COMMENTS = {
    "base_wrong_r3_correct": "r3 在 OOD normal 图像上优于 Base，是值得人工复核的正向泛化案例。",
    "base_correct_r3_wrong": "Base 正确而 r3 退化，提示 LingoQA 校准可能带来跨域损失。",
    "r3_normal_text_prior": "r3 在 normal 正确，但 text-only 也与答案高度重合，不能认定其依赖视觉。",
    "r3_normal_wrong_image_confound": "r3 在错误图像下仍高重合，显示跨域场景中的 wrong-image confound。",
    "r3_blank_high_f1": "空白图像下仍得到高 F1，是语言先验或答案格式偏置的重要样本。",
    "camera_specific_failure": "涉及相机视角的问题在 OOD 中失败，应检查 camera-label 对齐。",
    "object_token_failure": "涉及 object token 的问题失败，反映对象引用的跨数据集对齐困难。",
    "spatial_relation_failure": "空间关系问题失败，提示视觉几何推理泛化不足。",
    "counting_failure": "计数问题失败，需区分感知错误与语言先验回答。",
    "action_reasoning_failure": "动作推理问题失败，说明上层驾驶判断仍受域差异影响。",
}


def by_id(root: Path, model: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(group["normal"]["id"]): group for group in align_settings(load_model_settings(root, "drivelm", model, "strict_visual"))}


def scored(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}


def gap(scores: dict[str, dict[str, Any]]) -> float:
    return scores["normal"]["f1"] - max(scores[setting]["f1"] for setting in SETTINGS[1:])


def classify(capability: str, base: dict[str, dict[str, Any]], r3: dict[str, dict[str, Any]]) -> list[str]:
    types = []
    if r3["normal"]["f1"] >= base["normal"]["f1"] + 0.20:
        types.append("base_wrong_r3_correct")
    if base["normal"]["f1"] >= r3["normal"]["f1"] + 0.20:
        types.append("base_correct_r3_wrong")
    if r3["normal"]["f1"] >= 0.20 and r3["text_only"]["f1"] >= 0.20:
        types.append("r3_normal_text_prior")
    if r3["normal"]["f1"] >= 0.20 and r3["wrong_image"]["f1"] >= 0.20:
        types.append("r3_normal_wrong_image_confound")
    if r3["blank_image"]["f1"] >= 0.20:
        types.append("r3_blank_high_f1")
    if r3["normal"]["f1"] < 0.20 and f"{capability}_failure" in FAILURE_TYPES:
        types.append(f"{capability}_failure")
    return types


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 13 DriveLM OOD case gallery.")
    parser.add_argument("--prediction_root", default="outputs/predictions_drivelm_ood")
    parser.add_argument("--output_csv", default="outputs/final_report/drivelm_ood_case_gallery_100.csv")
    parser.add_argument("--output_html", default="outputs/final_report/drivelm_ood_case_gallery_100.html")
    parser.add_argument("--per_type", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.prediction_root)
    base_groups, r3_groups = by_id(root, BASE), by_id(root, R3)
    common = sorted(set(base_groups) & set(r3_groups))
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample_id in common:
        raw = r3_groups[sample_id]
        capability = classify_question(str(raw["normal"].get("question", "")))
        base, r3 = scored(base_groups[sample_id]), scored(raw)
        for failure_type in classify(capability, base, r3):
            selected[failure_type].append({
                "id": sample_id,
                "capability": capability,
                "question": raw["normal"].get("question", ""),
                "gold": raw["normal"].get("gold", ""),
                "image_paths": json.dumps(raw["normal"].get("image_paths", []), ensure_ascii=False),
                "image_labels": json.dumps(raw["normal"].get("image_labels", []), ensure_ascii=False),
                "normal_prediction": raw["normal"].get("prediction", ""),
                "text_only_prediction": raw["text_only"].get("prediction", ""),
                "wrong_image_prediction": raw["wrong_image"].get("prediction", ""),
                "blank_image_prediction": raw["blank_image"].get("prediction", ""),
                "normal_f1": r3["normal"]["f1"],
                "text_only_f1": r3["text_only"]["f1"],
                "wrong_image_f1": r3["wrong_image"]["f1"],
                "blank_image_f1": r3["blank_image"]["f1"],
                "case_gap": gap(r3),
                "failure_type": failure_type,
                "chinese_comment": COMMENTS[failure_type],
            })
    rows = []
    for failure_type in FAILURE_TYPES:
        candidates = selected.get(failure_type, [])
        candidates.sort(key=lambda row: (row["case_gap"], -row["normal_f1"]))
        rows.extend(candidates[: args.per_type])
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["id", "failure_type", "chinese_comment"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    chunks = [
        "<!doctype html><meta charset='utf-8'><title>DriveLM OOD Case Gallery</title>",
        "<style>body{font:14px Arial,sans-serif;margin:28px;color:#202124}h1,h2{margin-top:28px}"
        "table{border-collapse:collapse;width:100%;margin-bottom:22px}th,td{border:1px solid #ddd;"
        "padding:7px;vertical-align:top;max-width:360px;white-space:pre-wrap}th{background:#f3f5f7}"
        ".note{background:#eef5ff;padding:12px;border-left:4px solid #376fe5}</style>",
        "<h1>DriveLM OOD Strict Visual-Control Case Gallery</h1>",
        "<p class='note'>这是跨数据集 OOD 诊断结果，不是训练结果，也未用于构造训练数据。图片仅以路径与 camera labels 展示。</p>",
    ]
    for failure_type in FAILURE_TYPES:
        group = [row for row in rows if row["failure_type"] == failure_type]
        if not group:
            continue
        chunks.append(f"<h2>{html.escape(failure_type)}</h2><table><tr>")
        for field in fields:
            chunks.append(f"<th>{html.escape(field)}</th>")
        chunks.append("</tr>")
        for row in group:
            chunks.append("<tr>")
            for field in fields:
                chunks.append(f"<td>{html.escape(str(row.get(field, '')))}</td>")
            chunks.append("</tr>")
        chunks.append("</table>")
    Path(args.output_html).write_text("".join(chunks), encoding="utf-8")
    print(json.dumps({"cases": len(rows), "failure_type_counts": {key: len(selected.get(key, [])) for key in FAILURE_TYPES}, "csv": args.output_csv, "html": args.output_html}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
