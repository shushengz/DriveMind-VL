"""Compare two visual-control case files and explain per-case regressions.

The visual-control evaluator writes one JSONL row per sample with normal,
text-only, wrong-image, and blank-image F1 scores. This helper compares a
baseline run against a candidate run so we can tell whether a preference run
improved grounding or merely suppressed answers across all settings.
"""

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
SCORE_KEYS = ("normal", *CONTROL_KEYS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in read_jsonl(path)}


def score(row: dict[str, Any], key: str) -> float:
    return float(row.get(f"{key}_f1", 0.0) or 0.0)


def control_max(row: dict[str, Any]) -> float:
    return max(score(row, key) for key in CONTROL_KEYS)


def setting_gap(row: dict[str, Any]) -> float:
    return score(row, "normal") - control_max(row)


def answer_from_prediction(text: Any) -> str:
    parsed = parse_model_output(text)
    if parsed.get("ok") and isinstance(parsed.get("data"), dict):
        data = parsed["data"]
        return str(data.get("answer") or data.get("reason") or "")
    return str(text or "")


def short_text(text: Any, limit: int = 180) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\n", " ").replace("|", "/").strip()
    return value[:limit]


def classify_delta(row: dict[str, Any], normal_tol: float, control_tol: float) -> str:
    normal_delta = float(row["delta_normal_f1"])
    control_delta = float(row["delta_control_max_f1"])
    gap_delta = float(row["delta_setting_gap"])
    wrong_delta = float(row["delta_wrong_image_f1"])

    if normal_delta >= normal_tol and gap_delta >= 0:
        return "candidate_better_overall"
    if abs(normal_delta) < normal_tol and control_delta <= -control_tol and gap_delta >= 0:
        return "candidate_better_control_no_normal_loss"
    if normal_delta <= -normal_tol and control_delta <= -control_tol:
        return "answer_suppression_tradeoff"
    if normal_delta <= -normal_tol:
        return "normal_regression"
    if wrong_delta <= -control_tol and gap_delta > 0:
        return "wrong_image_control_improved"
    if gap_delta < 0:
        return "visual_gap_regression"
    return "mixed_or_neutral"


def build_rows(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    baseline_name: str,
    candidate_name: str,
    normal_tol: float,
    control_tol: float,
) -> list[dict[str, Any]]:
    common_ids = sorted(set(baseline) & set(candidate))
    rows: list[dict[str, Any]] = []
    for sample_id in common_ids:
        base = baseline[sample_id]
        cand = candidate[sample_id]
        gold = cand.get("gold") if isinstance(cand.get("gold"), dict) else {}
        row = {
            "id": sample_id,
            "capability": cand.get("capability") or base.get("capability", ""),
            "category": cand.get("category") or base.get("category", ""),
            "subcategory": cand.get("subcategory") or base.get("subcategory", ""),
            "gold_answer": short_text(gold.get("answer") or gold.get("reason") or ""),
            f"{baseline_name}_normal_f1": score(base, "normal"),
            f"{candidate_name}_normal_f1": score(cand, "normal"),
            "delta_normal_f1": round(score(cand, "normal") - score(base, "normal"), 4),
            f"{baseline_name}_text_only_f1": score(base, "text_only"),
            f"{candidate_name}_text_only_f1": score(cand, "text_only"),
            "delta_text_only_f1": round(score(cand, "text_only") - score(base, "text_only"), 4),
            f"{baseline_name}_wrong_image_f1": score(base, "wrong_image"),
            f"{candidate_name}_wrong_image_f1": score(cand, "wrong_image"),
            "delta_wrong_image_f1": round(score(cand, "wrong_image") - score(base, "wrong_image"), 4),
            f"{baseline_name}_blank_image_f1": score(base, "blank_image"),
            f"{candidate_name}_blank_image_f1": score(cand, "blank_image"),
            "delta_blank_image_f1": round(score(cand, "blank_image") - score(base, "blank_image"), 4),
            f"{baseline_name}_control_max_f1": control_max(base),
            f"{candidate_name}_control_max_f1": control_max(cand),
            "delta_control_max_f1": round(control_max(cand) - control_max(base), 4),
            f"{baseline_name}_setting_gap": setting_gap(base),
            f"{candidate_name}_setting_gap": setting_gap(cand),
            "delta_setting_gap": round(setting_gap(cand) - setting_gap(base), 4),
            f"{baseline_name}_normal_answer": short_text(answer_from_prediction(base.get("normal_prediction"))),
            f"{candidate_name}_normal_answer": short_text(answer_from_prediction(cand.get("normal_prediction"))),
        }
        row["delta_type"] = classify_delta(row, normal_tol=normal_tol, control_tol=control_tol)
        rows.append(row)
    return rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / len(rows), 4)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(row["delta_type"]) for row in rows)
    by_capability = defaultdict(Counter)
    for row in rows:
        by_capability[str(row.get("capability", ""))][str(row["delta_type"])] += 1
    return {
        "count": len(rows),
        "mean_delta_normal_f1": mean(rows, "delta_normal_f1"),
        "mean_delta_text_only_f1": mean(rows, "delta_text_only_f1"),
        "mean_delta_wrong_image_f1": mean(rows, "delta_wrong_image_f1"),
        "mean_delta_blank_image_f1": mean(rows, "delta_blank_image_f1"),
        "mean_delta_control_max_f1": mean(rows, "delta_control_max_f1"),
        "mean_delta_setting_gap": mean(rows, "delta_setting_gap"),
        "delta_type_counts": dict(by_type.most_common()),
        "delta_type_by_capability": {
            capability: dict(counter.most_common()) for capability, counter in sorted(by_capability.items())
        },
    }


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
        lines.append("| " + " | ".join(short_text(row.get(col, ""), 120) for col in columns) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    baseline_name: str,
    candidate_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normal_regressions = sorted(rows, key=lambda row: float(row["delta_normal_f1"]))[:12]
    control_improvements = sorted(rows, key=lambda row: float(row["delta_control_max_f1"]))[:12]
    gap_improvements = sorted(rows, key=lambda row: float(row["delta_setting_gap"]), reverse=True)[:12]
    good_tradeoffs = [
        row
        for row in sorted(rows, key=lambda item: (float(item["delta_setting_gap"]), -float(item["delta_normal_f1"])), reverse=True)
        if row["delta_type"] in {"candidate_better_control_no_normal_loss", "candidate_better_overall", "wrong_image_control_improved"}
    ][:12]

    columns = [
        "id",
        "capability",
        "delta_type",
        "delta_normal_f1",
        "delta_control_max_f1",
        "delta_setting_gap",
        "gold_answer",
        f"{baseline_name}_normal_answer",
        f"{candidate_name}_normal_answer",
    ]
    content = [
        f"# Visual-Control Case Comparison: {baseline_name} vs {candidate_name}",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Reading The Deltas",
        "",
        f"- Positive deltas mean `{candidate_name}` is higher than `{baseline_name}`.",
        "- Lower control F1 can be good only when normal F1 is preserved.",
        "- `answer_suppression_tradeoff` means both normal and controls dropped, so the model likely became less willing or less able to answer rather than better grounded.",
        "",
        "## Largest Normal Regressions",
        "",
        markdown_table(normal_regressions, columns),
        "",
        "## Largest Control Reductions",
        "",
        markdown_table(control_improvements, columns),
        "",
        "## Largest Setting-Gap Improvements",
        "",
        markdown_table(gap_improvements, columns),
        "",
        "## Best Tradeoff Candidates",
        "",
        markdown_table(good_tradeoffs, columns),
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare visual-control cases between two model runs.")
    parser.add_argument("--baseline_cases", required=True)
    parser.add_argument("--candidate_cases", required=True)
    parser.add_argument("--baseline_name", default="baseline")
    parser.add_argument("--candidate_name", default="candidate")
    parser.add_argument("--output_csv", default="outputs/cases/visual_control_case_comparison.csv")
    parser.add_argument("--output_json", default="outputs/eval_results/visual_control_case_comparison.json")
    parser.add_argument("--output_md", default="docs/visual_control_case_comparison.md")
    parser.add_argument("--normal_tol", type=float, default=0.05)
    parser.add_argument("--control_tol", type=float, default=0.05)
    args = parser.parse_args()

    rows = build_rows(
        index_rows(Path(args.baseline_cases)),
        index_rows(Path(args.candidate_cases)),
        baseline_name=args.baseline_name,
        candidate_name=args.candidate_name,
        normal_tol=args.normal_tol,
        control_tol=args.control_tol,
    )
    summary = summarize(rows)

    write_csv(Path(args.output_csv), rows)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(Path(args.output_md), rows, summary, args.baseline_name, args.candidate_name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
