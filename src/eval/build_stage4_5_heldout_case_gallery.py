"""Build an answer-only held-out case gallery from saved prediction files."""
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


def f1(group: dict[str, dict[str, Any]] | None, setting: str) -> float:
    return score_prediction(group[setting], "answer_only")["f1"] if group else 0.0


def classify(r3: dict[str, dict[str, Any]], sft2: dict[str, dict[str, Any]] | None, r2: dict[str, dict[str, Any]] | None) -> str:
    question = str(r3["normal"].get("question", ""))
    r3_n, r3_w, r3_t = (f1(r3, setting) for setting in ("normal", "wrong_image", "text_only"))
    blank = score_prediction(r3["blank_image"], "answer_only")
    if sft2 and r3_n > f1(sft2, "normal") + 0.2:
        return "heldout_r3_better_than_sft_v2"
    if sft2 and f1(sft2, "normal") > r3_n + 0.2:
        return "heldout_r3_worse_than_sft_v2"
    if r2 and r3_n > f1(r2, "normal") + 0.2:
        return "heldout_r3_better_than_r2"
    if r3_n >= 0.7 and r3_w >= r3_n - 0.05:
        return "r3_normal_ok_wrong_ok"
    if blank["hallucination"]:
        return "r3_blank_hallucination"
    if r3_t >= r3_n:
        return "r3_text_prior_bias"
    if SPATIAL.search(question) and r3_n < 0.5:
        return "r3_spatial_failure"
    if r3_n > max(r3_w, r3_t, f1(r3, "blank_image")):
        return "r3_visual_gain"
    return "unresolved"


def chinese_comment(case_type: str) -> str:
    comments = {
        "heldout_r3_better_than_sft_v2": "\u5728\u672a\u53c2\u4e0e\u8bad\u7ec3\u7684\u6837\u672c\u4e0a\uff0cr3 \u7684 normal \u56de\u7b54\u6bd4 SFT-v2 \u66f4\u63a5\u8fd1\u7b54\u6848\uff0c\u8fd9\u662f\u6709\u6548\u6cdb\u5316\u7684\u6b63\u5411\u4fe1\u53f7\u3002",
        "heldout_r3_worse_than_sft_v2": "\u5728 held-out \u6837\u672c\u4e0a SFT-v2 \u4f18\u4e8e r3\uff0c\u9700\u8981\u68c0\u67e5\u8f7b\u91cf\u6821\u51c6\u662f\u5426\u635f\u5bb3\u4e86 normal QA \u80fd\u529b\u3002",
        "heldout_r3_better_than_r2": "r3 \u5728 held-out normal \u95ee\u7b54\u4e0a\u4f18\u4e8e r2\uff0c\u8bf4\u660e answer-only \u548c\u66f4\u9ad8 normal replay \u6bd4\u4f8b\u53ef\u80fd\u66f4\u7a33\u5b9a\u3002",
        "r3_normal_ok_wrong_ok": "r3 \u5728 normal \u4e0b\u56de\u7b54\u6b63\u786e\uff0c\u4f46 wrong-image \u4e5f\u80fd\u63a5\u8fd1 gold\uff0c\u53ef\u80fd\u4ecd\u6709\u8bed\u8a00\u5148\u9a8c\u6216\u6570\u636e\u504f\u7f6e\u3002",
        "r3_blank_hallucination": "r3 \u5728 blank-image \u6761\u4ef6\u4e0b\u4ecd\u4f5c\u51fa\u5f3a\u89c6\u89c9\u65ad\u8a00\uff0c\u9700\u8981\u7ee7\u7eed\u964d\u4f4e control hallucination\u3002",
        "r3_text_prior_bias": "r3 \u5728 text-only \u4e0b\u4e0d\u4f9d\u8d56\u89c6\u89c9\u8bc1\u636e\u4e5f\u83b7\u5f97\u76f8\u5f53\u5f97\u5206\uff0c\u9700\u8981\u8b66\u60d5\u8bed\u8a00\u5148\u9a8c\u3002",
        "r3_spatial_failure": "\u8be5 held-out \u7a7a\u95f4\u5173\u7cfb\u95ee\u9898\u4e2d r3 normal \u56de\u7b54\u4e0d\u7a33\u5b9a\uff0c\u7a7a\u95f4 grounding \u4ecd\u9700\u589e\u5f3a\u3002",
        "r3_visual_gain": "\u8be5\u6837\u672c\u4e2d normal \u5f97\u5206\u9ad8\u4e8e\u6240\u6709 control \u8bbe\u7f6e\uff0c\u8868\u660e\u6a21\u578b\u4ece\u89c6\u89c9\u8bc1\u636e\u4e2d\u53d6\u5f97\u4e86\u6709\u6548\u6536\u76ca\u3002",
    }
    return comments.get(case_type, "\u8be5\u6837\u672c\u9700\u8981\u7ed3\u5408 raw prediction \u7ee7\u7eed\u5ba1\u67e5\u3002")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["message"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["message"]
    out = ["<html><head><meta charset='utf-8'><style>table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;max-width:360px}</style></head><body><h1>Stage 4.5 Held-out Case Gallery</h1><table><tr>"]
    out += [f"<th>{html.escape(field)}</th>" for field in fields] + ["</tr>"]
    for row in rows:
        out.append("<tr>")
        out += [f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields]
        out.append("</tr>")
    out.append("</table></body></html>")
    path.write_text("".join(out), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build held-out Stage 4.5 case gallery.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--output_csv", default="outputs/final_report/stage4_5_heldout_case_gallery.csv")
    parser.add_argument("--output_html", default="outputs/final_report/stage4_5_heldout_case_gallery.html")
    parser.add_argument("--limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.prediction_root)
    try:
        r3 = load_aligned(root, "sft_v3_r3_lingo_smoke")
    except Exception:
        message = "Held-out r3 predictions are not available. Build a non-overlapping eval pool before generating the gallery."
        rows = [{"message": message}]
        write_csv(Path(args.output_csv), rows)
        write_html(Path(args.output_html), rows)
        print(json.dumps({"count": 0, "message": message}, ensure_ascii=False, indent=2))
        return
    try:
        sft2 = load_aligned(root, "sft_v2")
    except Exception:
        sft2 = {}
    try:
        r2 = load_aligned(root, "sft_v3_r2_lingo_smoke")
    except Exception:
        r2 = {}
    rows: list[dict[str, Any]] = []
    for sample_id, group in r3.items():
        kind = classify(group, sft2.get(sample_id), r2.get(sample_id))
        scored = {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}
        rows.append({
            "id": sample_id,
            "case_type": kind,
            "question": group["normal"].get("question", ""),
            "gold": group["normal"].get("gold", ""),
            "normal_f1": scored["normal"]["f1"],
            "text_only_f1": scored["text_only"]["f1"],
            "wrong_image_f1": scored["wrong_image"]["f1"],
            "blank_image_f1": scored["blank_image"]["f1"],
            "case_gap": scored["normal"]["f1"] - max(scored[s]["f1"] for s in ("text_only", "wrong_image", "blank_image")),
            "normal_prediction": group["normal"].get("prediction", ""),
            "wrong_image_prediction": group["wrong_image"].get("prediction", ""),
            "blank_image_prediction": group["blank_image"].get("prediction", ""),
            "text_only_prediction": group["text_only"].get("prediction", ""),
            "chinese_comment": chinese_comment(kind),
        })
    rows.sort(key=lambda row: (row["case_type"] == "unresolved", row["case_gap"]))
    rows = rows[: args.limit]
    write_csv(Path(args.output_csv), rows)
    write_html(Path(args.output_html), rows)
    print(json.dumps({"count": len(rows), "csv": args.output_csv, "html": args.output_html}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
