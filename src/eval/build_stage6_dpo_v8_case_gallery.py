"""Build the held-out DPO-v8 versus r3 case gallery from raw predictions."""
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

SPATIAL = re.compile(r"left|right|front|back|lane|traffic light|pedestrian|vehicle|behind|ahead", re.I)


def load_aligned(root: Path, model: str) -> dict[str, dict[str, dict[str, Any]]]:
    aligned = align_settings(load_model_settings(root, "lingoqa", model, "strict_visual"))
    return {str(group["normal"].get("id")): group for group in aligned}


def scored(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}


def case_gap(values: dict[str, dict[str, Any]]) -> float:
    return values["normal"]["f1"] - max(values[setting]["f1"] for setting in ("text_only", "wrong_image", "blank_image"))


def classify(
    group: dict[str, dict[str, Any]],
    dpo: dict[str, dict[str, Any]],
    r3_values: dict[str, dict[str, Any]],
    dpo7_values: dict[str, dict[str, Any]] | None,
) -> str:
    if any(r3_values[setting]["hallucination"] and not dpo[setting]["hallucination"] for setting in ("text_only", "wrong_image", "blank_image")):
        return "r3_hallucination_dpo_v8_fixed"
    if r3_values["normal"]["f1"] > dpo["normal"]["f1"] + 0.2:
        return "r3_correct_dpo_v8_wrong"
    if dpo["normal"]["f1"] >= 0.7 and dpo["wrong_image"]["f1"] >= dpo["normal"]["f1"] - 0.05:
        return "dpo_v8_normal_ok_wrong_ok"
    if dpo["blank_image"]["hallucination"]:
        return "dpo_v8_blank_hallucination"
    if dpo["text_only"]["f1"] >= dpo["normal"]["f1"]:
        return "dpo_v8_text_prior_bias"
    if dpo["normal"]["refusal"]:
        return "dpo_v8_over_refusal"
    if case_gap(dpo) > case_gap(r3_values) + 0.2:
        return "dpo_v8_case_gap_improved"
    if dpo7_values and dpo["normal"]["f1"] > dpo7_values["normal"]["f1"] + 0.2:
        return "dpo_v8_better_than_dpo_v7"
    if SPATIAL.search(str(group["normal"].get("question", ""))) and dpo["normal"]["f1"] < 0.5:
        return "dpo_v8_spatial_failure"
    return "unresolved"


def comment(kind: str) -> str:
    comments = {
        "r3_hallucination_dpo_v8_fixed": "r3 在控制条件下产生了不可靠断言，而 DPO-v8 改为更谨慎的回答；这是 Preference-v8 预期改善的直接案例。",
        "r3_correct_dpo_v8_wrong": "r3 在 normal 条件下表现更好，而 DPO-v8 退化，提示偏好校准可能损伤正常问答能力。",
        "dpo_v8_normal_ok_wrong_ok": "DPO-v8 在真实输入下回答较好，但 wrong-image 条件仍接近 gold，仍需警惕语言先验或错误证据混淆。",
        "dpo_v8_blank_hallucination": "DPO-v8 在 blank-image 条件下仍出现强断言，说明控制组幻觉未完全消除。",
        "dpo_v8_text_prior_bias": "DPO-v8 在 text-only 下得分不低于 normal，说明仍可能依赖语言先验而非视觉证据。",
        "dpo_v8_over_refusal": "DPO-v8 在 normal 输入下出现拒答，需要警惕 control preference 过强导致过度校准。",
        "dpo_v8_case_gap_improved": "DPO-v8 在该样本上拉开了 normal 与 control 的差距，体现了更好的视觉依赖性。",
        "dpo_v8_better_than_dpo_v7": "DPO-v8 在 held-out normal 条件下明显优于 DPO-v7，是新初始化与偏好设计的正向信号。",
        "dpo_v8_spatial_failure": "该空间关系问题在 DPO-v8 下仍不稳定，需继续保护 spatial anchor。",
    }
    return comments.get(kind, "该 held-out 样本未归入优先错误类型，保留用于人工复核 raw prediction。")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["message"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["message"]
    content = ["<html><head><meta charset='utf-8'><style>table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;max-width:360px}</style></head><body><h1>Stage 6 DPO-v8 Held-out Case Gallery</h1><table><tr>"]
    content += [f"<th>{html.escape(field)}</th>" for field in fields] + ["</tr>"]
    for row in rows:
        content.append("<tr>")
        content += [f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields]
        content.append("</tr>")
    content.append("</table></body></html>")
    path.write_text("".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage 6 held-out DPO-v8 case gallery.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--output_csv", default="outputs/final_report/stage6_dpo_v8_case_gallery.csv")
    parser.add_argument("--output_html", default="outputs/final_report/stage6_dpo_v8_case_gallery.html")
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.prediction_root)
    try:
        r3 = load_aligned(root, "sft_v3_r3_lingo_smoke")
        dpo8 = load_aligned(root, "dpo_v8_lingo_smoke")
    except Exception as exc:
        rows = [{"message": f"DPO-v8 held-out predictions are not available: {exc}"}]
        write_csv(Path(args.output_csv), rows)
        write_html(Path(args.output_html), rows)
        print(json.dumps({"count": 0, "message": rows[0]["message"]}, ensure_ascii=False, indent=2))
        return
    try:
        dpo7 = load_aligned(root, "dpo_v7")
    except Exception:
        dpo7 = {}
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(set(r3) & set(dpo8)):
        r3_values = scored(r3[sample_id])
        dpo_values = scored(dpo8[sample_id])
        dpo7_values = scored(dpo7[sample_id]) if sample_id in dpo7 else None
        kind = classify(dpo8[sample_id], dpo_values, r3_values, dpo7_values)
        rows.append({
            "id": sample_id,
            "case_type": kind,
            "question": dpo8[sample_id]["normal"].get("question", ""),
            "gold": dpo8[sample_id]["normal"].get("gold", ""),
            "r3_normal_f1": r3_values["normal"]["f1"],
            "dpo_v8_normal_f1": dpo_values["normal"]["f1"],
            "r3_case_gap": case_gap(r3_values),
            "dpo_v8_case_gap": case_gap(dpo_values),
            "normal_prediction": dpo8[sample_id]["normal"].get("prediction", ""),
            "text_only_prediction": dpo8[sample_id]["text_only"].get("prediction", ""),
            "wrong_image_prediction": dpo8[sample_id]["wrong_image"].get("prediction", ""),
            "blank_image_prediction": dpo8[sample_id]["blank_image"].get("prediction", ""),
            "chinese_comment": comment(kind),
        })
    rows.sort(key=lambda row: (row["case_type"] == "unresolved", row["dpo_v8_case_gap"] - row["r3_case_gap"]))
    rows = rows[: args.limit]
    write_csv(Path(args.output_csv), rows)
    write_html(Path(args.output_html), rows)
    print(json.dumps({"count": len(rows), "csv": args.output_csv, "html": args.output_html}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
