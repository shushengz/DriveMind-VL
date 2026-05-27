"""Analyze visual-control failure patterns from existing prediction outputs."""

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


CONTROL_KEYS = ("text_only", "wrong_image", "blank_image")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def index_predictions(path: str) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return {str(row.get("id")): row for row in read_jsonl(file_path)}


def parse_answer(text: Any) -> str:
    parsed = parse_model_output(text)
    if parsed.get("ok") and isinstance(parsed.get("data"), dict):
        data = parsed["data"]
        return str(data.get("answer") or data.get("reason") or "")
    return str(text or "")


def short(text: Any, limit: int = 180) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\n", " ").replace("|", "/").strip()
    return value[:limit]


def score(row: dict[str, Any], key: str) -> float:
    return float(row.get(f"{key}_f1", 0.0) or 0.0)


def control_max(row: dict[str, Any]) -> float:
    return max(score(row, key) for key in CONTROL_KEYS)


def best_control(row: dict[str, Any]) -> str:
    return max(CONTROL_KEYS, key=lambda key: score(row, key))


def gold_answer(row: dict[str, Any]) -> str:
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    return str(gold.get("answer") or gold.get("reason") or "")


def prediction_answer(index: dict[str, dict[str, Any]], sample_id: str) -> str:
    row = index.get(sample_id, {})
    return parse_answer(row.get("prediction", ""))


def classify(row: dict[str, Any], tol: float, low_f1: float) -> str:
    normal = score(row, "normal")
    text = score(row, "text_only")
    wrong = score(row, "wrong_image")
    blank = score(row, "blank_image")
    control = max(text, wrong, blank)
    if control > normal + tol:
        if blank >= text and blank >= wrong:
            return "blank_beats_normal"
        if text >= wrong:
            return "text_beats_normal"
        return "wrong_image_beats_normal"
    if abs(wrong - normal) <= tol and normal <= low_f1:
        return "wrong_image_invariant_low"
    if abs(wrong - normal) <= tol:
        return "wrong_image_invariant"
    if normal <= low_f1:
        return "normal_low"
    return "acceptable_or_mixed"


def build_rows(
    cases: list[dict[str, Any]],
    normal_index: dict[str, dict[str, Any]],
    text_index: dict[str, dict[str, Any]],
    wrong_index: dict[str, dict[str, Any]],
    blank_index: dict[str, dict[str, Any]],
    tol: float,
    low_f1: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        sample_id = str(case.get("id") or "")
        normal = score(case, "normal")
        control = control_max(case)
        mode = classify(case, tol=tol, low_f1=low_f1)
        rows.append(
            {
                "id": sample_id,
                "failure_type": mode,
                "capability": case.get("capability", ""),
                "category": case.get("category", ""),
                "normal_f1": normal,
                "text_only_f1": score(case, "text_only"),
                "wrong_image_f1": score(case, "wrong_image"),
                "blank_image_f1": score(case, "blank_image"),
                "control_max_f1": control,
                "visual_dependency_gap": round(normal - control, 4),
                "best_control": best_control(case),
                "gold_answer": short(gold_answer(case), 240),
                "normal_answer": short(
                    prediction_answer(normal_index, sample_id) or parse_answer(case.get("normal_prediction", "")),
                    240,
                ),
                "text_only_answer": short(prediction_answer(text_index, sample_id), 240),
                "wrong_image_answer": short(prediction_answer(wrong_index, sample_id), 240),
                "blank_image_answer": short(prediction_answer(blank_index, sample_id), 240),
            }
        )
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows), 4)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row["failure_type"]) for row in rows)
    by_capability = defaultdict(Counter)
    by_category = defaultdict(Counter)
    for row in rows:
        by_capability[str(row.get("capability", "unknown"))][str(row["failure_type"])] += 1
        by_category[str(row.get("category", "unknown"))][str(row["failure_type"])] += 1
    return {
        "count": len(rows),
        "mean_normal_f1": mean(rows, "normal_f1"),
        "mean_text_only_f1": mean(rows, "text_only_f1"),
        "mean_wrong_image_f1": mean(rows, "wrong_image_f1"),
        "mean_blank_image_f1": mean(rows, "blank_image_f1"),
        "mean_visual_dependency_gap": mean(rows, "visual_dependency_gap"),
        "failure_type_counts": dict(by_type.most_common()),
        "failure_type_by_capability": {
            key: dict(counter.most_common()) for key, counter in sorted(by_capability.items())
        },
        "failure_type_by_category": {
            key: dict(counter.most_common()) for key, counter in sorted(by_category.items())
        },
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_None._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(short(row.get(col, ""), 120) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any], title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    worst_gap = sorted(rows, key=lambda row: float(row["visual_dependency_gap"]))[:15]
    blank_beats = [row for row in rows if row["failure_type"] == "blank_beats_normal"][:15]
    text_beats = [row for row in rows if row["failure_type"] == "text_beats_normal"][:15]
    invariant_low = [row for row in rows if row["failure_type"] == "wrong_image_invariant_low"][:15]
    columns = [
        "id",
        "failure_type",
        "capability",
        "category",
        "normal_f1",
        "control_max_f1",
        "visual_dependency_gap",
        "gold_answer",
        "normal_answer",
        "blank_image_answer",
    ]
    content = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Worst Visual Dependency Gaps",
        "",
        markdown_table(worst_gap, columns),
        "",
        "## Blank Image Beats Normal",
        "",
        markdown_table(blank_beats, columns),
        "",
        "## Text Only Beats Normal",
        "",
        markdown_table(text_beats, columns),
        "",
        "## Wrong Image Invariant And Low",
        "",
        markdown_table(invariant_low, columns),
        "",
        "## Interpretation",
        "",
        "- `blank_beats_normal` and `text_beats_normal` indicate language-prior shortcuts.",
        "- `wrong_image_invariant_low` indicates the model output barely changes when images are swapped, while normal accuracy is also weak.",
        "- These groups should be prioritized for DriveLM-aligned SFT or preference data instead of continuing LingoQA-only preference tuning.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze visual-control failure modes from existing cases.")
    parser.add_argument("--cases", required=True, help="*_visual_control_cases.jsonl")
    parser.add_argument("--normal_predictions", default="")
    parser.add_argument("--text_only_predictions", default="")
    parser.add_argument("--wrong_image_predictions", default="")
    parser.add_argument("--blank_image_predictions", default="")
    parser.add_argument("--output_json", default="outputs/eval_results/visual_control_failure_analysis.json")
    parser.add_argument("--output_csv", default="outputs/eval_results/visual_control_failure_cases.csv")
    parser.add_argument("--output_md", default="docs/visual_control_failure_analysis.md")
    parser.add_argument("--title", default="Visual-Control Failure Analysis")
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--low_f1", type=float, default=0.25)
    args = parser.parse_args()

    cases = read_jsonl(Path(args.cases))
    rows = build_rows(
        cases,
        index_predictions(args.normal_predictions),
        index_predictions(args.text_only_predictions),
        index_predictions(args.wrong_image_predictions),
        index_predictions(args.blank_image_predictions),
        tol=args.tol,
        low_f1=args.low_f1,
    )
    summary = summarize(rows)
    write_json(Path(args.output_json), {"summary": summary, "cases": rows})
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows, summary, args.title)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_json}, {args.output_csv}, {args.output_md}")


if __name__ == "__main__":
    main()
