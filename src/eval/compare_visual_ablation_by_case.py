"""Compare visual-ablation predictions per sample and capability."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.output_parser import parse_model_output
from src.eval.eval_external_vqa_breakdown import infer_capability, summarize
from src.eval.refusal_detection import is_refusal_prediction
from src.eval.run_all_eval import load_jsonl, token_f1, write_jsonl


SETTING_ORDER = ("normal", "text_only", "wrong_image", "blank_image")
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "is",
    "are",
    "be",
    "this",
    "that",
    "there",
    "vehicle",
    "car",
    "ego",
}


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def score_prediction(row: dict[str, Any]) -> float:
    parsed = parse_model_output(row.get("prediction"))
    pred = parsed["data"] if parsed["ok"] else {}
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    pred_text = str(pred.get("answer") or pred.get("reason") or row.get("prediction") or "")
    gold_text = str(gold.get("answer") or gold.get("reason") or "")
    return token_f1(pred_text, gold_text)


def prediction_text(row: dict[str, Any]) -> str:
    parsed = parse_model_output(row.get("prediction"))
    if parsed["ok"] and isinstance(parsed.get("data"), dict):
        data = parsed["data"]
        return str(data.get("answer") or "") + " " + str(data.get("reason") or "")
    return str(row.get("prediction") or "")


def tokens(text: Any) -> set[str]:
    value = "".join(ch.lower() if ch.isalnum() else " " for ch in str(text or ""))
    return {token for token in value.split() if len(token) >= 3 and token not in STOPWORDS}


def visual_terms(row: dict[str, Any]) -> set[str]:
    perception = row.get("perception") if isinstance(row.get("perception"), dict) else {}
    objects = perception.get("objects") if isinstance(perception.get("objects"), (dict, list)) else {}
    terms: set[str] = set()
    if isinstance(objects, dict):
        iterable = objects.values()
    else:
        iterable = objects
    for item in iterable:
        if not isinstance(item, dict):
            continue
        for key in ("Category", "category", "Status", "status", "Visual_description", "visual_description", "description"):
            terms.update(tokens(item.get(key, "")))
    terms.update(tokens(perception.get("scene", "")))
    terms.update(tokens(perception.get("risk_hint", "")))
    return terms


def visual_term_overlap(row: dict[str, Any]) -> float:
    terms = visual_terms(row)
    if not terms:
        return 0.0
    pred_terms = tokens(prediction_text(row))
    return round(len(terms & pred_terms) / len(terms), 4)


def build_case_rows(indexed: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    case_rows: list[dict[str, Any]] = []
    ids = sorted(set.intersection(*(set(rows.keys()) for rows in indexed.values())))
    for sample_id in ids:
        normal_row = indexed["normal"][sample_id]
        scores = {setting: score_prediction(indexed[setting][sample_id]) for setting in SETTING_ORDER}
        refusals = {setting: is_refusal_prediction(indexed[setting][sample_id].get("prediction")) for setting in SETTING_ORDER}
        overlaps = {setting: visual_term_overlap(indexed[setting][sample_id]) for setting in SETTING_ORDER}
        control_max = max(scores["text_only"], scores["wrong_image"], scores["blank_image"])
        case_rows.append(
            {
                "id": sample_id,
                "capability": infer_capability(normal_row),
                "category": (normal_row.get("gold") or {}).get("category", "unknown"),
                "subcategory": (normal_row.get("gold") or {}).get("subcategory", "unknown"),
                "normal_f1": round(scores["normal"], 4),
                "text_only_f1": round(scores["text_only"], 4),
                "wrong_image_f1": round(scores["wrong_image"], 4),
                "blank_image_f1": round(scores["blank_image"], 4),
                "control_max_f1": round(control_max, 4),
                "visual_dependency_gap": round(scores["normal"] - control_max, 4),
                "normal_visual_term_overlap": overlaps["normal"],
                "text_only_visual_term_overlap": overlaps["text_only"],
                "wrong_image_visual_term_overlap": overlaps["wrong_image"],
                "blank_image_visual_term_overlap": overlaps["blank_image"],
                "normal_refusal": refusals["normal"],
                "text_only_refusal": refusals["text_only"],
                "wrong_image_refusal": refusals["wrong_image"],
                "blank_image_refusal": refusals["blank_image"],
                "gold": normal_row.get("gold", {}),
                "normal_prediction": normal_row.get("prediction"),
            }
        )
    return case_rows


def summarize_gap(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_rows:
        return {"count": 0, "normal_f1": 0.0, "control_max_f1": 0.0, "visual_dependency_gap": 0.0}
    count = len(case_rows)
    return {
        "count": count,
        "normal_f1": round(sum(float(row["normal_f1"]) for row in case_rows) / count, 4),
        "text_only_f1": round(sum(float(row["text_only_f1"]) for row in case_rows) / count, 4),
        "wrong_image_f1": round(sum(float(row["wrong_image_f1"]) for row in case_rows) / count, 4),
        "blank_image_f1": round(sum(float(row["blank_image_f1"]) for row in case_rows) / count, 4),
        "control_max_f1": round(sum(float(row["control_max_f1"]) for row in case_rows) / count, 4),
        "visual_dependency_gap": round(sum(float(row["visual_dependency_gap"]) for row in case_rows) / count, 4),
        "normal_visual_term_overlap": round(
            sum(float(row.get("normal_visual_term_overlap", 0.0)) for row in case_rows) / count, 4
        ),
        "text_only_visual_term_overlap": round(
            sum(float(row.get("text_only_visual_term_overlap", 0.0)) for row in case_rows) / count, 4
        ),
        "wrong_image_visual_term_overlap": round(
            sum(float(row.get("wrong_image_visual_term_overlap", 0.0)) for row in case_rows) / count, 4
        ),
        "blank_image_visual_term_overlap": round(
            sum(float(row.get("blank_image_visual_term_overlap", 0.0)) for row in case_rows) / count, 4
        ),
        "positive_gap_rate": round(sum(1 for row in case_rows if float(row["visual_dependency_gap"]) > 0) / count, 4),
        "normal_refusal_rate": round(sum(1 for row in case_rows if row.get("normal_refusal")) / count, 4),
        "text_only_refusal_rate": round(sum(1 for row in case_rows if row.get("text_only_refusal")) / count, 4),
        "wrong_image_refusal_rate": round(sum(1 for row in case_rows if row.get("wrong_image_refusal")) / count, 4),
        "blank_image_refusal_rate": round(sum(1 for row in case_rows if row.get("blank_image_refusal")) / count, 4),
    }


def build_report(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        grouped[str(row["capability"])].append(row)
    return {
        "overall": summarize_gap(case_rows),
        "by_capability": {name: summarize_gap(rows) for name, rows in sorted(grouped.items())},
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_gap_table(groups: dict[str, Any]) -> None:
    print("+------------------------------+-------+----------+-------------+----------+")
    print("| group                        | count | normal   | control_max | gap      |")
    print("+------------------------------+-------+----------+-------------+----------+")
    for name, metrics in groups.items():
        print(
            f"| {name[:28]:<28} | {int(metrics['count']):<5} | {metrics['normal_f1']:<8.4f} | "
            f"{metrics['control_max_f1']:<11.4f} | {metrics['visual_dependency_gap']:<8.4f} |"
        )
    print("+------------------------------+-------+----------+-------------+----------+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare visual-ablation external VQA predictions by sample.")
    parser.add_argument("--normal", required=True)
    parser.add_argument("--text_only", required=True)
    parser.add_argument("--wrong_image", required=True)
    parser.add_argument("--blank_image", required=True)
    parser.add_argument("--output", default="outputs/eval_results/intelli_visual_ablation_case_summary.json")
    parser.add_argument("--cases_output", default="outputs/cases/intelli_visual_ablation_cases.jsonl")
    args = parser.parse_args()

    indexed = {
        "normal": index_rows(Path(args.normal)),
        "text_only": index_rows(Path(args.text_only)),
        "wrong_image": index_rows(Path(args.wrong_image)),
        "blank_image": index_rows(Path(args.blank_image)),
    }
    case_rows = build_case_rows(indexed)
    report = build_report(case_rows)
    write_json(Path(args.output), report)
    write_jsonl(Path(args.cases_output), case_rows)

    print("== Overall ==")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print("\n== By Capability ==")
    print_gap_table(report["by_capability"])
    print(f"\nwrote visual-ablation case summary to {args.output}")
    print(f"wrote {len(case_rows)} visual-ablation cases to {args.cases_output}")


if __name__ == "__main__":
    main()
