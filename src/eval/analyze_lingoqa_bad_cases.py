"""Build a reviewable bad-case taxonomy for LingoQA visual-control runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.output_parser import parse_model_output
from src.eval.eval_external_vqa_breakdown import infer_capability
from src.eval.run_all_eval import load_jsonl, token_f1


SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def answer_text(row: dict[str, Any]) -> str:
    parsed = parse_model_output(row.get("prediction"))
    if parsed.get("ok") and isinstance(parsed.get("data"), dict):
        data = parsed["data"]
        return str(data.get("answer") or data.get("reason") or "")
    return str(row.get("prediction") or "")


def normalize_short_answer(text: str) -> str:
    return " ".join(text.lower().replace(".", "").replace(",", "").split())


def is_possible_metric_artifact(prediction: str, gold: str, f1: float) -> bool:
    pred = normalize_short_answer(prediction)
    ref = normalize_short_answer(gold)
    if f1 > 0:
        return False
    numeric_aliases = {
        "0": {"zero", "none", "no"},
        "1": {"one"},
        "2": {"two"},
        "3": {"three"},
    }
    if pred == ref:
        return True
    if pred in numeric_aliases and ref in numeric_aliases[pred]:
        return True
    if ref in numeric_aliases and pred in numeric_aliases[ref]:
        return True
    yes_aliases = {"yes", "yeah", "true"}
    no_aliases = {"no", "false"}
    return (pred in yes_aliases and ref in yes_aliases) or (pred in no_aliases and ref in no_aliases)


def classify_case(
    capability: str,
    scores: dict[str, float],
    answers: dict[str, str],
    gold_text: str,
    pass_threshold: float,
    positive_margin: float,
) -> tuple[str, list[str], str]:
    normal = scores["normal"]
    control_scores = {name: scores[name] for name in ("text_only", "wrong_image", "blank_image")}
    control_max_name, control_max = max(control_scores.items(), key=lambda item: item[1])
    flags: list[str] = []

    if normal >= control_max + positive_margin:
        primary = "visual_helped"
        recommendation = "Keep as positive grounding evidence; inspect visual evidence wording."
    elif normal < pass_threshold and control_max < pass_threshold:
        primary = "all_settings_failed"
        recommendation = "Review whether the question requires finer visual evidence or has ambiguous labels."
    elif control_max_name == "text_only" and scores["text_only"] >= normal:
        primary = "language_prior_or_question_bias"
        recommendation = "Add visual-evidence requirement and verify if the question can be answered from priors."
    elif control_max_name == "wrong_image" and scores["wrong_image"] >= normal:
        primary = "wrong_image_confound"
        recommendation = "Check if the model is answering from language priors or if references are too generic."
    elif control_max_name == "blank_image" and scores["blank_image"] >= normal:
        primary = "blank_image_confound"
        recommendation = "Strengthen prompt against hallucinated visual evidence and inspect answer ambiguity."
    else:
        primary = "visually_unstable"
        recommendation = "Needs manual review; compare normal and control rationales."

    if capability == "spatial_localization" and normal < pass_threshold:
        flags.append("spatial_failure")
    if capability == "reasoning_world_knowledge" and normal < pass_threshold:
        flags.append("reasoning_failure")
    if capability == "object_recognition" and normal < pass_threshold:
        flags.append("object_recognition_failure")
    if capability == "counting" and normal < pass_threshold:
        flags.append("counting_failure")
    if is_possible_metric_artifact(answers["normal"], gold_text, normal):
        flags.append("possible_metric_artifact")
    if scores["wrong_image"] >= scores["normal"] and scores["wrong_image"] >= pass_threshold:
        flags.append("wrong_image_beats_or_ties_normal")
    if scores["text_only"] >= scores["normal"] and scores["text_only"] >= pass_threshold:
        flags.append("text_only_beats_or_ties_normal")

    return primary, flags, recommendation


def build_taxonomy(
    indexed: dict[str, dict[str, dict[str, Any]]],
    dataset_by_id: dict[str, dict[str, Any]],
    pass_threshold: float,
    positive_margin: float,
) -> list[dict[str, Any]]:
    common_ids = sorted(set.intersection(*(set(rows.keys()) for rows in indexed.values())))
    rows: list[dict[str, Any]] = []
    for sample_id in common_ids:
        normal_row = indexed["normal"][sample_id]
        source_row = dataset_by_id.get(sample_id, {})
        gold = normal_row.get("gold") if isinstance(normal_row.get("gold"), dict) else {}
        gold_text = str(gold.get("answer") or gold.get("reason") or "")
        answers = {name: answer_text(indexed[name][sample_id]) for name in SETTINGS}
        scores = {name: token_f1(answers[name], gold_text) for name in SETTINGS}
        capability = infer_capability(normal_row)
        primary, flags, recommendation = classify_case(
            capability=capability,
            scores=scores,
            answers=answers,
            gold_text=gold_text,
            pass_threshold=pass_threshold,
            positive_margin=positive_margin,
        )
        external = normal_row.get("meta", {}).get("external", {})
        source_external = source_row.get("meta", {}).get("external", {}) if isinstance(source_row.get("meta"), dict) else {}
        rows.append(
            {
                "id": sample_id,
                "capability": capability,
                "question": source_row.get("instruction") or normal_row.get("instruction", ""),
                "image": source_row.get("image", normal_row.get("media", {}).get("image", "")),
                "num_images": len(source_external.get("image_paths", [])) if isinstance(source_external, dict) else "",
                "gold_answer": gold_text,
                "normal_answer": answers["normal"],
                "text_only_answer": answers["text_only"],
                "wrong_image_answer": answers["wrong_image"],
                "blank_image_answer": answers["blank_image"],
                "normal_f1": round(scores["normal"], 4),
                "text_only_f1": round(scores["text_only"], 4),
                "wrong_image_f1": round(scores["wrong_image"], 4),
                "blank_image_f1": round(scores["blank_image"], 4),
                "control_max_f1": round(max(scores[name] for name in ("text_only", "wrong_image", "blank_image")), 4),
                "visual_dependency_gap": round(
                    scores["normal"] - max(scores[name] for name in ("text_only", "wrong_image", "blank_image")), 4
                ),
                "primary_error_type": primary,
                "secondary_flags": ";".join(flags),
                "recommendation": recommendation,
                "category": gold.get("category", external.get("category", "")),
                "subcategory": gold.get("subcategory", external.get("subcategory", "")),
                "source": normal_row.get("meta", {}).get("benchmark_source", external.get("benchmark_source", "")),
                "reviewer_note": "",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(row["primary_error_type"] for row in rows)
    by_capability = Counter(row["capability"] for row in rows)
    type_by_capability: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        type_by_capability[row["capability"]][row["primary_error_type"]] += 1
    return {
        "count": len(rows),
        "by_primary_error_type": dict(by_type.most_common()),
        "by_capability": dict(by_capability.most_common()),
        "primary_error_type_by_capability": {
            capability: dict(counter.most_common()) for capability, counter in sorted(type_by_capability.items())
        },
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = [str(row.get(col, "")).replace("\n", " ")[:140] for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hardest = sorted(rows, key=lambda row: (float(row["normal_f1"]), float(row["visual_dependency_gap"])))[:12]
    confounds = sorted(rows, key=lambda row: float(row["visual_dependency_gap"]))[:12]
    positive = sorted(rows, key=lambda row: float(row["visual_dependency_gap"]), reverse=True)[:12]
    content = [
        "# LingoQA Bad Case Taxonomy",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Interpretation",
        "",
        "- `visual_helped`: normal image/frame input beats all controls by the configured margin.",
        "- `language_prior_or_question_bias`: text-only matches or beats normal, suggesting the question may be answerable without vision.",
        "- `wrong_image_confound`: wrong-frame matches or beats normal, a strong warning against claiming stable visual grounding.",
        "- `blank_image_confound`: blank image matches or beats normal, often caused by priors or answer/reference ambiguity.",
        "- `all_settings_failed`: none of the settings reaches the pass threshold; these are the best candidates for manual review and data repair.",
        "",
        "## Hardest Cases",
        "",
        markdown_table(hardest, ["id", "capability", "normal_f1", "primary_error_type", "secondary_flags", "question", "gold_answer", "normal_answer"]),
        "",
        "## Strongest Control Confounds",
        "",
        markdown_table(confounds, ["id", "capability", "visual_dependency_gap", "primary_error_type", "text_only_f1", "wrong_image_f1", "blank_image_f1", "question"]),
        "",
        "## Positive Grounding Evidence",
        "",
        markdown_table(positive, ["id", "capability", "visual_dependency_gap", "normal_f1", "control_max_f1", "question", "gold_answer", "normal_answer"]),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze LingoQA bad cases from visual-control predictions.")
    parser.add_argument("--normal", required=True)
    parser.add_argument("--text_only", required=True)
    parser.add_argument("--wrong_image", required=True)
    parser.add_argument("--blank_image", required=True)
    parser.add_argument("--dataset", default="", help="Optional original dataset JSONL used to restore instruction/image metadata.")
    parser.add_argument("--csv_output", default="outputs/cases/lingoqa_bad_case_taxonomy.csv")
    parser.add_argument("--json_output", default="outputs/eval_results/lingoqa_bad_case_taxonomy_summary.json")
    parser.add_argument("--report_output", default="docs/lingoqa_bad_case_analysis.md")
    parser.add_argument("--pass_threshold", type=float, default=0.2)
    parser.add_argument("--positive_margin", type=float, default=0.05)
    args = parser.parse_args()

    indexed = {
        "normal": index_rows(Path(args.normal)),
        "text_only": index_rows(Path(args.text_only)),
        "wrong_image": index_rows(Path(args.wrong_image)),
        "blank_image": index_rows(Path(args.blank_image)),
    }
    dataset_by_id = index_rows(Path(args.dataset)) if args.dataset else {}
    rows = build_taxonomy(indexed, dataset_by_id, args.pass_threshold, args.positive_margin)
    summary = summarize(rows)
    write_csv(Path(args.csv_output), rows)
    write_json(Path(args.json_output), summary)
    write_report(Path(args.report_output), rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote taxonomy CSV to {args.csv_output}")
    print(f"wrote taxonomy summary to {args.json_output}")
    print(f"wrote taxonomy report to {args.report_output}")


if __name__ == "__main__":
    main()
