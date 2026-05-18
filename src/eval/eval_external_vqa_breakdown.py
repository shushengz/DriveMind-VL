"""Break down external VQA metrics by capability and benchmark metadata."""

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
from src.data.external_vqa_taxonomy import infer_external_vqa_capability
from src.eval.run_all_eval import load_jsonl, token_f1
from src.rewards.total_reward import total_reward


def text_blob(row: dict[str, Any], include_view: bool = False) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    fields = [
        row.get("id", ""),
        row.get("instruction", ""),
        row.get("gold", {}).get("category", "") if isinstance(row.get("gold"), dict) else "",
        row.get("gold", {}).get("subcategory", "") if isinstance(row.get("gold"), dict) else "",
        row.get("gold", {}).get("answer", "") if isinstance(row.get("gold"), dict) else "",
        external.get("category", ""),
        external.get("subcategory", ""),
    ]
    if include_view:
        fields.append(external.get("shooting_angle", ""))
    return " ".join(str(field).lower() for field in fields if field is not None)


def infer_capability(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    return infer_external_vqa_capability(
        instruction=str(row.get("instruction") or ""),
        category=str(gold.get("category") or external.get("category") or ""),
        subcategory=str(gold.get("subcategory") or external.get("subcategory") or ""),
        reference=str(gold.get("answer") or gold.get("reason") or ""),
    )


def get_external_value(row: dict[str, Any], key: str) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    if key == "capability":
        return infer_capability(row)
    if key == "source":
        return str(meta.get("benchmark_source") or external.get("benchmark_source") or meta.get("source") or "unknown")
    if key == "category":
        gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
        return str(gold.get("category") or external.get("category") or "unknown")
    if key == "subcategory":
        gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
        return str(gold.get("subcategory") or external.get("subcategory") or "unknown")
    if key == "shooting_angle":
        return str(external.get("shooting_angle") or "unknown")
    if key == "weather":
        vehicle_state = row.get("vehicle_state") if isinstance(row.get("vehicle_state"), dict) else {}
        return str(vehicle_state.get("weather") or "unknown")
    raise ValueError(f"unsupported group key: {key}")


def score_row(row: dict[str, Any], pass_threshold: float) -> dict[str, Any]:
    parsed = parse_model_output(row.get("prediction"))
    pred = parsed["data"] if parsed["ok"] else {}
    gold = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    pred_text = str(pred.get("answer") or pred.get("reason") or row.get("prediction") or "")
    gold_text = str(gold.get("answer") or gold.get("reason") or "")
    f1 = token_f1(pred_text, gold_text)
    reward = total_reward(row.get("prediction"), gold, row.get("vehicle_state"), row.get("meta"))
    return {
        "id": row.get("id"),
        "json_valid": 1.0 if parsed["ok"] else 0.0,
        "f1": f1,
        "pass": 1.0 if f1 >= pass_threshold else 0.0,
        "reward": reward.get("total", 0.0),
        "capability": infer_capability(row),
        "prediction_text": pred_text,
        "gold_text": gold_text,
    }


def summarize(scored_rows: list[dict[str, Any]]) -> dict[str, float]:
    if not scored_rows:
        return {"count": 0, "json_validity": 0.0, "external_answer_f1": 0.0, "pass_rate": 0.0, "avg_reward": 0.0}
    count = len(scored_rows)
    return {
        "count": count,
        "json_validity": round(sum(row["json_valid"] for row in scored_rows) / count, 4),
        "external_answer_f1": round(sum(row["f1"] for row in scored_rows) / count, 4),
        "pass_rate": round(sum(row["pass"] for row in scored_rows) / count, 4),
        "avg_reward": round(sum(row["reward"] for row in scored_rows) / count, 4),
    }


def build_report(rows: list[dict[str, Any]], pass_threshold: float) -> dict[str, Any]:
    external_rows = [row for row in rows if row.get("meta", {}).get("task_type") == "external_vqa"]
    scored_by_id = {str(row.get("id")): score_row(row, pass_threshold) for row in external_rows}
    report: dict[str, Any] = {
        "overall": summarize(list(scored_by_id.values())),
        "by_capability": {},
        "by_category": {},
        "by_subcategory": {},
        "by_shooting_angle": {},
        "by_weather": {},
        "cases": scored_by_id,
    }
    for section, key in (
        ("by_capability", "capability"),
        ("by_category", "category"),
        ("by_subcategory", "subcategory"),
        ("by_shooting_angle", "shooting_angle"),
        ("by_weather", "weather"),
    ):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in external_rows:
            grouped[get_external_value(row, key)].append(scored_by_id[str(row.get("id"))])
        report[section] = {name: summarize(group) for name, group in sorted(grouped.items())}
    return report


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def print_table(title: str, groups: dict[str, dict[str, float]]) -> None:
    print(f"\n== {title} ==")
    print("+------------------------------+-------+----------+----------+------------+")
    print("| group                        | count | f1       | pass     | avg_reward |")
    print("+------------------------------+-------+----------+----------+------------+")
    for name, metrics in groups.items():
        print(
            f"| {name[:28]:<28} | {int(metrics['count']):<5} | "
            f"{metrics['external_answer_f1']:<8.4f} | {metrics['pass_rate']:<8.4f} | "
            f"{metrics['avg_reward']:<10.4f} |"
        )
    print("+------------------------------+-------+----------+----------+------------+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Break down external VQA metrics by capability and metadata.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", default="outputs/eval_results/external_vqa_breakdown.json")
    parser.add_argument("--pass_threshold", type=float, default=0.2)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions))
    report = build_report(rows, args.pass_threshold)
    write_json(Path(args.output), report)
    print("== Overall ==")
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    print_table("By Capability", report["by_capability"])
    print_table("By Category", report["by_category"])
    print(f"\nwrote external VQA breakdown to {args.output}")


if __name__ == "__main__":
    main()
