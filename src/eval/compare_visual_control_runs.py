"""Compare multiple visual-control evaluation runs.

This is a no-GPU reporting helper. It reads the per-setting visual-control
summary and optional per-case summary produced by the DriveMind-VL eval scripts,
then writes compact Markdown/CSV/JSON tables for experiment tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SETTING_KEYS = ("normal", "text_only", "wrong_image", "blank_image")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_run_spec(text: str) -> tuple[str, Path, Path | None]:
    if "=" in text and (":" not in text or text.index("=") < text.index(":")):
        name_part, path_part = text.split("=", 1)
        parts = [name_part, *path_part.split(":", 1)]
    else:
        parts = text.split(":", 2)
    if len(parts) < 2:
        raise ValueError("--run must be NAME:SUMMARY_JSON[:CASE_SUMMARY_JSON] or NAME=SUMMARY_JSON[:CASE_SUMMARY_JSON]")
    name = parts[0].strip()
    summary = Path(parts[1].strip())
    case_summary = Path(parts[2].strip()) if len(parts) == 3 and parts[2].strip() else infer_case_summary_path(summary)
    if not name:
        raise ValueError("run name cannot be empty")
    return name, summary, case_summary


def infer_case_summary_path(summary_path: Path) -> Path | None:
    text = summary_path.as_posix()
    suffix = "_visual_control_summary.json"
    if text.endswith(suffix):
        return Path(text[: -len(suffix)] + "_visual_control_case_summary.json")
    return None


def metric(data: dict[str, Any], path: tuple[str, ...], default: float = 0.0) -> float:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


def setting_f1(summary: dict[str, Any], setting: str) -> float:
    return metric(summary, (setting, "external_answer_f1"))


def setting_gap(summary: dict[str, Any]) -> float:
    if "_visual_dependency" in summary:
        return metric(summary, ("_visual_dependency", "visual_dependency_gap"))
    normal = setting_f1(summary, "normal")
    control_max = max(setting_f1(summary, key) for key in SETTING_KEYS if key != "normal")
    return normal - control_max


def load_run(name: str, summary_path: Path, case_summary_path: Path | None) -> dict[str, Any]:
    summary = read_json(summary_path)
    case_summary: dict[str, Any] = {}
    if case_summary_path and case_summary_path.exists():
        case_summary = read_json(case_summary_path)
    overall_case = case_summary.get("overall", {}) if isinstance(case_summary.get("overall"), dict) else {}

    row = {
        "name": name,
        "summary_path": summary_path.as_posix(),
        "case_summary_path": case_summary_path.as_posix() if case_summary_path else "",
        "normal_f1": setting_f1(summary, "normal"),
        "text_only_f1": setting_f1(summary, "text_only"),
        "wrong_image_f1": setting_f1(summary, "wrong_image"),
        "blank_image_f1": setting_f1(summary, "blank_image"),
        "setting_visual_gap": setting_gap(summary),
        "json_validity_normal": metric(summary, ("normal", "json_validity")),
        "schema_completeness_normal": metric(summary, ("normal", "schema_completeness")),
        "avg_reward_normal": metric(summary, ("normal", "avg_reward")),
        "per_case_control_max_f1": float(overall_case.get("control_max_f1", 0.0) or 0.0),
        "per_case_visual_gap": float(overall_case.get("visual_dependency_gap", 0.0) or 0.0),
        "positive_gap_rate": float(overall_case.get("positive_gap_rate", 0.0) or 0.0),
        "normal_refusal_rate": float(overall_case.get("normal_refusal_rate", 0.0) or 0.0),
        "control_refusal_rate_max": max(
            float(overall_case.get("text_only_refusal_rate", 0.0) or 0.0),
            float(overall_case.get("wrong_image_refusal_rate", 0.0) or 0.0),
            float(overall_case.get("blank_image_refusal_rate", 0.0) or 0.0),
        ),
    }
    return row


def status_for(
    row: dict[str, Any],
    min_normal_f1: float,
    min_per_case_gap: float,
    baseline_name: str,
    incumbent_name: str,
    incumbent: dict[str, Any] | None,
) -> str:
    if row["name"] == baseline_name:
        return "baseline"
    if row["name"] == incumbent_name:
        return "incumbent"
    if row["normal_f1"] < min_normal_f1:
        return "reject_normal_regression"
    if row["per_case_visual_gap"] < min_per_case_gap:
        return "diagnostic_only_gap_weak"
    if row["setting_visual_gap"] <= 0:
        return "diagnostic_only_setting_gap"
    if incumbent is not None:
        if row["normal_f1"] < incumbent["normal_f1"] or row["setting_visual_gap"] < incumbent["setting_visual_gap"]:
            return "reject_incumbent_regression"
    return "candidate"


def delta_rows(rows: list[dict[str, Any]], baseline_name: str, incumbent_name: str) -> list[dict[str, Any]]:
    baseline = next((row for row in rows if row["name"] == baseline_name), rows[0] if rows else None)
    incumbent = next((row for row in rows if row["name"] == incumbent_name), None) if incumbent_name else None
    if baseline is None:
        return rows
    for row in rows:
        row["delta_normal_f1_vs_baseline"] = row["normal_f1"] - baseline["normal_f1"]
        row["delta_setting_gap_vs_baseline"] = row["setting_visual_gap"] - baseline["setting_visual_gap"]
        row["delta_per_case_gap_vs_baseline"] = row["per_case_visual_gap"] - baseline["per_case_visual_gap"]
        row["delta_positive_gap_rate_vs_baseline"] = row["positive_gap_rate"] - baseline["positive_gap_rate"]
        if incumbent is not None:
            row["delta_normal_f1_vs_incumbent"] = row["normal_f1"] - incumbent["normal_f1"]
            row["delta_setting_gap_vs_incumbent"] = row["setting_visual_gap"] - incumbent["setting_visual_gap"]
            row["delta_per_case_gap_vs_incumbent"] = row["per_case_visual_gap"] - incumbent["per_case_visual_gap"]
    return rows


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"runs": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown(path: Path, rows: list[dict[str, Any]], title: str, baseline_name: str, incumbent_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("name", "run"),
        ("status", "status"),
        ("normal_f1", "normal"),
        ("text_only_f1", "text"),
        ("wrong_image_f1", "wrong"),
        ("blank_image_f1", "blank"),
        ("setting_visual_gap", "setting_gap"),
        ("per_case_visual_gap", "case_gap"),
        ("positive_gap_rate", "pos_rate"),
        ("delta_normal_f1_vs_baseline", "d_normal"),
        ("delta_per_case_gap_vs_baseline", "d_case_gap"),
    ]
    if incumbent_name:
        columns.extend(
            [
                ("delta_normal_f1_vs_incumbent", "d_normal_inc"),
                ("delta_setting_gap_vs_incumbent", "d_setting_inc"),
                ("delta_per_case_gap_vs_incumbent", "d_case_inc"),
            ]
        )
    lines = [f"# {title}", ""]
    lines.append(f"Baseline: `{baseline_name}`")
    if incumbent_name:
        lines.append(f"Incumbent: `{incumbent_name}`")
    lines.append("")
    lines.append("| " + " | ".join(label for _, label in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key, "")) for key, _ in columns) + " |")
    lines.append("")
    lines.append("Decision rule used by this report:")
    lines.append("")
    lines.append("- `baseline`: reference run for deltas.")
    lines.append("- `incumbent`: current best local run that new checkpoints should beat.")
    lines.append("- `candidate`: normal F1 and visual gaps pass the configured floors.")
    lines.append("- `diagnostic_only_gap_weak`: normal F1 is acceptable, but per-case visual dependency is still weak.")
    lines.append("- `reject_normal_regression`: normal-image quality regressed below the configured floor.")
    lines.append("- `reject_incumbent_regression`: acceptable versus baseline, but weaker than the incumbent on normal F1 or setting-level visual gap.")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare visual-control experiment runs.")
    parser.add_argument(
        "--run",
        action="append",
        help="Run spec: NAME:SUMMARY_JSON[:CASE_SUMMARY_JSON]. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=[],
        help="Compatibility alias: NAME=SUMMARY_JSON[:CASE_SUMMARY_JSON] repeated after --runs.",
    )
    parser.add_argument("--baseline", default="", help="Baseline run name. Defaults to the first run.")
    parser.add_argument("--incumbent", default="", help="Current best run name that new candidates should beat.")
    parser.add_argument("--output_json", default="outputs/eval_results/visual_control_run_comparison.json")
    parser.add_argument("--output_csv", default="outputs/eval_results/visual_control_run_comparison.csv")
    parser.add_argument("--output_md", default="docs/visual_control_run_comparison.md")
    parser.add_argument("--title", default="LingoQA Visual-Control Run Comparison")
    parser.add_argument("--min_normal_f1", type=float, default=0.35)
    parser.add_argument("--min_per_case_gap", type=float, default=-0.01)
    parser.add_argument("--allow_missing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    run_specs = list(args.run or []) + list(args.runs or [])
    if not run_specs:
        raise SystemExit("at least one --run or --runs entry is required")
    for spec in run_specs:
        name, summary_path, case_summary_path = parse_run_spec(spec)
        if not summary_path.exists():
            if args.allow_missing:
                print(f"skip missing run {name}: {summary_path}")
                continue
            raise FileNotFoundError(summary_path)
        row = load_run(name, summary_path, case_summary_path)
        rows.append(row)

    if not rows:
        raise SystemExit("no runs loaded")

    baseline_name = args.baseline or rows[0]["name"]
    incumbent = next((row for row in rows if row["name"] == args.incumbent), None) if args.incumbent else None
    for row in rows:
        row["status"] = status_for(row, args.min_normal_f1, args.min_per_case_gap, baseline_name, args.incumbent, incumbent)
    rows = delta_rows(rows, baseline_name, args.incumbent)
    write_json(Path(args.output_json), rows)
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows, args.title, baseline_name, args.incumbent)

    print(f"loaded runs: {len(rows)}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
