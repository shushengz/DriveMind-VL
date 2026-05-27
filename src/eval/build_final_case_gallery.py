"""Build a grouped final gallery from held-out saved predictions, without inference."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rescore_answer_only import SETTINGS, align_settings, load_model_settings, score_prediction

MODELS = {
    "Base": "base_qwen25vl_3b",
    "SFT-v2": "sft_v2",
    "r3": "sft_v3_r3_lingo_smoke",
    "DPO-v8": "dpo_v8_lingo_smoke",
    "DPO-v8.1": "dpo_v8_1_lingo_smoke_step50",
    "DPO-v8.2": "dpo_v8_2_lingo_smoke_step25",
}
TYPES = [
    ("r3_vs_base_sftv2_improvement", "r3 相比 Base/SFT-v2 明显改进", "r3 展示了正常视觉问答能力提升，是主模型选择的正向证据。"),
    ("r3_visual_gain", "r3 visual gain", "normal 与受控输入之间的差距扩大，说明该样本更依赖视觉证据。"),
    ("r3_control_overlap", "r3 normal 正确但 control 也高重合", "即便最终模型也会在控制条件下猜到高重合答案，说明仍有语言先验风险。"),
    ("dpo_v8_tradeoff", "DPO-v8 修正 hallucination 但 case_gap 变差", "DPO 的显式幻觉修正不等同于 case-level 视觉依赖提升。"),
    ("dpo_v81_blank_prior", "DPO-v8.1 修正后 blank prior 仍存在", "model-mined preference 仍未抑制空白图像下的高重合先验回答。"),
    ("dpo_v82_blank_unresolved", "DPO-v8.2 未解决 blank high-F1", "针对性 preference 仍未消除关键失败模式。"),
    ("text_only_prior_bias", "text-only prior bias", "没有图像时仍出现答案重合，暴露语言先验捷径。"),
    ("wrong_image_confound", "wrong-image confound", "错误图像没有可靠降低答案匹配，暴露图像证据混淆。"),
    ("blank_high_f1_prior", "blank-image high-F1 prior answer", "空白输入仍高 F1，是本项目最关键的失败诊断之一。"),
    ("normal_answer_regression", "normal answer regression", "校准模型在正常输入上退化，显示过度校准的代价。"),
]


def load_groups(root: Path, internal: str) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(group["normal"]["id"]): group for group in align_settings(load_model_settings(root, "lingoqa", internal, "strict_visual"))}


def scores(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}


def gap(value: dict[str, dict[str, Any]]) -> float:
    return value["normal"]["f1"] - max(value[setting]["f1"] for setting in ("text_only", "wrong_image", "blank_image"))


def control_max(value: dict[str, dict[str, Any]]) -> float:
    return max(value[setting]["f1"] for setting in ("text_only", "wrong_image", "blank_image"))


def rankers() -> dict[str, Callable[[dict[str, dict[str, dict[str, Any]]]], float]]:
    return {
        "r3_vs_base_sftv2_improvement": lambda s: s["r3"]["normal"]["f1"] - max(s["Base"]["normal"]["f1"], s["SFT-v2"]["normal"]["f1"]),
        "r3_visual_gain": lambda s: gap(s["r3"]) - gap(s["Base"]),
        "r3_control_overlap": lambda s: s["r3"]["normal"]["f1"] + control_max(s["r3"]),
        "dpo_v8_tradeoff": lambda s: (s["r3"]["blank_image"]["f1"] - s["DPO-v8"]["blank_image"]["f1"]) + (gap(s["r3"]) - gap(s["DPO-v8"])),
        "dpo_v81_blank_prior": lambda s: s["DPO-v8.1"]["blank_image"]["f1"] + (s["r3"]["normal"]["f1"] - s["DPO-v8.1"]["normal"]["f1"]),
        "dpo_v82_blank_unresolved": lambda s: s["DPO-v8.2"]["blank_image"]["f1"],
        "text_only_prior_bias": lambda s: s["DPO-v8.2"]["text_only"]["f1"],
        "wrong_image_confound": lambda s: s["DPO-v8.2"]["wrong_image"]["f1"],
        "blank_high_f1_prior": lambda s: max(s["r3"]["blank_image"]["f1"], s["DPO-v8.2"]["blank_image"]["f1"]),
        "normal_answer_regression": lambda s: s["r3"]["normal"]["f1"] - s["DPO-v8.2"]["normal"]["f1"],
    }


def predicates() -> dict[str, Callable[[dict[str, dict[str, dict[str, Any]]]], bool]]:
    control = ("text_only", "wrong_image", "blank_image")
    return {
        "r3_vs_base_sftv2_improvement": lambda s: s["r3"]["normal"]["f1"] > max(s["Base"]["normal"]["f1"], s["SFT-v2"]["normal"]["f1"]),
        "r3_visual_gain": lambda s: gap(s["r3"]) > gap(s["Base"]),
        "r3_control_overlap": lambda s: s["r3"]["normal"]["f1"] >= 0.20 and control_max(s["r3"]) >= 0.20,
        "dpo_v8_tradeoff": lambda s: any(s["r3"][setting]["hallucination"] and not s["DPO-v8"][setting]["hallucination"] for setting in control) and gap(s["DPO-v8"]) < gap(s["r3"]),
        "dpo_v81_blank_prior": lambda s: any(s["r3"][setting]["hallucination"] and not s["DPO-v8.1"][setting]["hallucination"] for setting in control) and s["DPO-v8.1"]["blank_image"]["f1"] >= 0.20,
        "dpo_v82_blank_unresolved": lambda s: s["DPO-v8.2"]["blank_image"]["f1"] >= 0.20,
        "text_only_prior_bias": lambda s: s["DPO-v8.2"]["text_only"]["f1"] >= 0.20,
        "wrong_image_confound": lambda s: s["DPO-v8.2"]["wrong_image"]["f1"] >= 0.20,
        "blank_high_f1_prior": lambda s: max(s["r3"]["blank_image"]["f1"], s["DPO-v8.2"]["blank_image"]["f1"]) >= 0.20,
        "normal_answer_regression": lambda s: s["r3"]["normal"]["f1"] > s["DPO-v8.2"]["normal"]["f1"] + 0.10,
    }


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def output_row(sample_id: str, failure_type: str, title: str, comment: str, groups: dict[str, dict[str, dict[str, Any]]], scored: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    normal = groups["r3"][sample_id]["normal"]
    outputs = {name: {setting: groups[name][sample_id][setting].get("prediction", "") for setting in SETTINGS} for name in MODELS}
    normal_f1 = {name: round(scored[name]["normal"]["f1"], 4) for name in MODELS}
    control_f1 = {name: {setting: round(scored[name][setting]["f1"], 4) for setting in ("text_only", "wrong_image", "blank_image")} for name in MODELS}
    gaps = {name: round(gap(scored[name]), 4) for name in MODELS}
    return {
        "id": sample_id,
        "dataset": "lingoqa",
        "question": normal.get("question", ""),
        "gold": normal.get("gold", ""),
        "image_paths": json_compact(normal.get("image_paths", [])),
        "model_outputs": json_compact(outputs),
        "normal_f1_by_model": json_compact(normal_f1),
        "control_f1_by_model": json_compact(control_f1),
        "case_gap_by_model": json_compact(gaps),
        "failure_type": title,
        "why_important": failure_type,
        "chinese_comment": comment,
    }


def write_html(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["failure_type"], []).append(row)
    chunks = ["""<html><head><meta charset="utf-8"><title>DriveMind-VL Final Case Gallery</title>
<style>body{font-family:Arial,sans-serif;margin:28px;color:#222}h1{margin-bottom:6px}h2{margin-top:32px;border-bottom:1px solid #ddd;padding-bottom:8px}.case{border:1px solid #ddd;border-radius:6px;padding:14px;margin:12px 0}.label{font-weight:bold;color:#334}.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}pre{white-space:pre-wrap;word-break:break-word;background:#f6f7f8;padding:8px;font-size:12px}</style></head><body><h1>DriveMind-VL Final Case Gallery</h1><p>Held-out strict visual-control, answer-only scoring. 图片路径保留为文本，报告不依赖图像文件渲染。</p>"""]
    for failure_type, cases in grouped.items():
        chunks.append(f"<h2>{html.escape(failure_type)}</h2>")
        for row in cases:
            outputs = json.loads(row["model_outputs"])
            chunks.append(f"<div class='case'><div class='label'>{html.escape(row['id'])}</div><p><b>Q:</b> {html.escape(row['question'])}<br><b>Gold:</b> {html.escape(row['gold'])}</p><p>{html.escape(row['chinese_comment'])}</p><p><b>Image paths:</b> {html.escape(row['image_paths'])}</p><div class='grid'>")
            for name, settings in outputs.items():
                content = "\n".join(f"{setting}: {settings.get(setting, '')}" for setting in SETTINGS)
                chunks.append(f"<div><b>{html.escape(name)}</b><pre>{html.escape(content)}</pre></div>")
            chunks.append(f"</div><p><b>normal_f1:</b> {html.escape(row['normal_f1_by_model'])}<br><b>control_f1:</b> {html.escape(row['control_f1_by_model'])}<br><b>case_gap:</b> {html.escape(row['case_gap_by_model'])}</p></div>")
    chunks.append("</body></html>")
    path.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final held-out case gallery without inference.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--output_csv", default="outputs/final_report/final_case_gallery.csv")
    parser.add_argument("--output_html", default="outputs/final_report/final_case_gallery.html")
    parser.add_argument("--per_type", type=int, default=2)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    root = Path(args.prediction_root)
    by_model = {name: load_groups(root, internal) for name, internal in MODELS.items()}
    ids = sorted(set.intersection(*(set(value) for value in by_model.values())))
    scored = {sample_id: {name: scores(by_model[name][sample_id]) for name in MODELS} for sample_id in ids}
    rank = rankers()
    eligible = predicates()
    rows = []
    for failure_type, title, comment in TYPES:
        candidates = [sample_id for sample_id in ids if eligible[failure_type](scored[sample_id])]
        if len(candidates) < args.per_type:
            raise ValueError(f"not enough evidence cases for {failure_type}: found {len(candidates)}, need {args.per_type}")
        selected = sorted(candidates, key=lambda sample_id: rank[failure_type](scored[sample_id]), reverse=True)[: args.per_type]
        rows.extend(output_row(sample_id, failure_type, title, comment, by_model, scored[sample_id]) for sample_id in selected)
    if args.dry_run:
        print(json.dumps({"common_ids": len(ids), "gallery_rows": len(rows), "types": len(TYPES)}, ensure_ascii=False, indent=2))
        return
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_html(Path(args.output_html), rows)
    print(json.dumps({"common_ids": len(ids), "gallery_rows": len(rows), "output_html": args.output_html}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
