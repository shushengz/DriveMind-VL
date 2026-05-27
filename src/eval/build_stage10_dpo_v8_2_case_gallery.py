"""Build Stage 10 held-out case gallery comparing r3, v8.1 and v8.2."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rescore_answer_only import SETTINGS, align_settings, load_model_settings, score_prediction

R3 = "sft_v3_r3_lingo_smoke"
V81 = "dpo_v8_1_lingo_smoke_step50"
V82 = "dpo_v8_2_lingo_smoke_step25"


def groups(root: Path, model: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(group["normal"]["id"]): group for group in align_settings(load_model_settings(root, "lingoqa", model, "strict_visual"))}


def scores(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}


def gap(scored: dict[str, dict[str, Any]]) -> float:
    return scored["normal"]["f1"] - max(scored[s]["f1"] for s in ("text_only", "wrong_image", "blank_image"))


def classify(r3: dict[str, dict[str, Any]], v81: dict[str, dict[str, Any]], v82: dict[str, dict[str, Any]]) -> str:
    if r3["blank_image"]["f1"] >= 0.20 and v82["blank_image"]["f1"] < 0.20:
        return "r3_blank_high_f1_v82_fixed"
    if v81["blank_image"]["f1"] >= 0.20 and v82["blank_image"]["f1"] < 0.20:
        return "v81_blank_unfixed_v82_fixed"
    if r3["normal"]["f1"] > v82["normal"]["f1"] + 0.20:
        return "r3_normal_correct_v82_wrong"
    if v82["normal"]["f1"] >= 0.50 and any(v82[s]["f1"] >= 0.20 for s in ("text_only", "wrong_image", "blank_image")):
        return "v82_normal_ok_control_also_high"
    if v82["normal"]["refusal"]:
        return "v82_over_refusal"
    if gap(v82) > gap(r3) + 0.10:
        return "v82_case_gap_improved"
    if gap(v82) < gap(r3) - 0.10:
        return "v82_case_gap_regressed"
    return "review_other"


COMMENTS = {
    "r3_blank_high_f1_v82_fixed": "r3 在空白图像下仍给出与答案高重合的先验回答，v8.2 已将其压回谨慎回答方向。",
    "v81_blank_unfixed_v82_fixed": "v8.1 未解决该空白图像先验重合，v8.2 的 case-gap-aware pair 对此样本有效。",
    "r3_normal_correct_v82_wrong": "r3 正常视觉问答更好，v8.2 在该样本损失了正常识别能力，需要计入代价。",
    "v82_normal_ok_control_also_high": "v8.2 的 normal 回答可用，但控制输入仍与 gold 高重合，视觉依赖尚不可靠。",
    "v82_over_refusal": "v8.2 在正常图像上拒答，属于校准过强导致的正常能力损伤。",
    "v82_case_gap_improved": "v8.2 扩大了 normal 与控制条件的差距，是目标行为的正向样本。",
    "v82_case_gap_regressed": "v8.2 让该样本的 case gap 变差，应检查是否仍受先验答案或过度拒答影响。",
    "review_other": "该样本未落入首要模式，保留供人工复查 raw prediction。",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["message"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["message"]
    chunks = ["<html><head><meta charset='utf-8'><style>table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:6px;vertical-align:top;max-width:360px}</style></head><body><h1>Stage 10 DPO-v8.2 Case Gallery</h1><table><tr>"]
    chunks.extend(f"<th>{html.escape(field)}</th>" for field in fields)
    chunks.append("</tr>")
    for row in rows:
        chunks.append("<tr>")
        chunks.extend(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        chunks.append("</tr>")
    chunks.append("</table></body></html>")
    path.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 10 DPO-v8.2 held-out case gallery.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--output_csv", default="outputs/final_report/stage10_dpo_v8_2_case_gallery.csv")
    parser.add_argument("--output_html", default="outputs/final_report/stage10_dpo_v8_2_case_gallery.html")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    by_model = {name: groups(Path(args.prediction_root), name) for name in (R3, V81, V82)}
    rows = []
    for sample_id in sorted(set.intersection(*(set(items) for items in by_model.values()))):
        r3, v81, v82 = (scores(by_model[name][sample_id]) for name in (R3, V81, V82))
        kind = classify(r3, v81, v82)
        group = by_model[V82][sample_id]
        rows.append({
            "id": sample_id,
            "case_type": kind,
            "question": group["normal"].get("question", ""),
            "gold": group["normal"].get("gold", ""),
            "r3_normal_f1": r3["normal"]["f1"],
            "v81_normal_f1": v81["normal"]["f1"],
            "v82_normal_f1": v82["normal"]["f1"],
            "r3_case_gap": gap(r3),
            "v82_case_gap": gap(v82),
            "r3_blank_f1": r3["blank_image"]["f1"],
            "v81_blank_f1": v81["blank_image"]["f1"],
            "v82_blank_f1": v82["blank_image"]["f1"],
            "v82_normal_prediction": group["normal"].get("prediction", ""),
            "v82_text_only_prediction": group["text_only"].get("prediction", ""),
            "v82_wrong_image_prediction": group["wrong_image"].get("prediction", ""),
            "v82_blank_image_prediction": group["blank_image"].get("prediction", ""),
            "chinese_comment": COMMENTS[kind],
        })
    priority = list(COMMENTS)
    rows.sort(key=lambda row: (priority.index(row["case_type"]), row["v82_case_gap"] - row["r3_case_gap"]))
    rows = rows[: args.limit]
    write_csv(Path(args.output_csv), rows)
    write_html(Path(args.output_html), rows)
    print(json.dumps({"count": len(rows), "csv": args.output_csv, "html": args.output_html}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
