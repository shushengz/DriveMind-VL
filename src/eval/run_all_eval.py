"""Run all local MVP evaluation metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.agent.output_parser import parse_model_output
    from src.eval.eval_json_validity import json_validity
    from src.eval.eval_risk import risk_accuracy
    from src.eval.eval_tool_call import tool_accuracy
    from src.eval.eval_safety import unsafe_rejection_rate
    from src.rewards.total_reward import total_reward
except Exception:
    from agent.output_parser import parse_model_output
    from eval_json_validity import json_validity
    from eval_risk import risk_accuracy
    from eval_tool_call import tool_accuracy
    from eval_safety import unsafe_rejection_rate
    from rewards.total_reward import total_reward


REQUIRED_BY_TASK = {
    "risk_reasoning": {"task", "risk_level", "risk_object", "reason", "suggestion"},
    "tool_call": {"task", "tool", "arguments", "reason"},
    "safety_rejection": {"task", "tool", "arguments", "refusal", "reason", "safe_alternative"},
    "cabin_understanding": {"task", "driver_state", "passenger_state", "reason", "suggestion"},
    "personalized_service": {"task", "tool", "arguments", "reason"},
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                if line.strip():
                    rows.append(json.loads(line))
            except Exception as exc:
                print(f"skip line {line_no}: {exc}")
    return rows


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    rewards = [
        total_reward(row.get("prediction"), row.get("gold"), row.get("vehicle_state"), row.get("meta"))["total"]
        for row in rows
    ]
    schema_scores = [schema_completeness(row) for row in rows]
    reason_scores = [reason_keyword_hit(row) for row in rows]
    return {
        "json_validity": round(json_validity(rows), 4),
        "risk_accuracy": round(risk_accuracy(rows), 4),
        "tool_accuracy": round(tool_accuracy(rows), 4),
        "tool_argument_accuracy": round(tool_argument_accuracy(rows), 4),
        "unsafe_rejection_rate": round(unsafe_rejection_rate(rows), 4),
        "schema_completeness": round(sum(schema_scores) / len(schema_scores), 4) if schema_scores else 0.0,
        "reason_keyword_hit": round(sum(reason_scores) / len(reason_scores), 4) if reason_scores else 0.0,
        "avg_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
    }


def schema_completeness(row: dict[str, Any]) -> float:
    parsed = parse_model_output(row.get("prediction"))
    if not parsed["ok"]:
        return 0.0
    task = row.get("meta", {}).get("task_type") or parsed["data"].get("task")
    required = REQUIRED_BY_TASK.get(task, set())
    if not required:
        return 1.0
    present = sum(1 for field in required if field in parsed["data"])
    return present / len(required)


def reason_keyword_hit(row: dict[str, Any]) -> float:
    parsed = parse_model_output(row.get("prediction"))
    if not parsed["ok"]:
        return 0.0
    pred = parsed["data"]
    reason = str(pred.get("reason", "")).lower()
    if not reason:
        return 0.0
    gold = row.get("gold", {})
    vehicle_state = row.get("vehicle_state", {})
    keywords = [
        str(gold.get("risk_object", "")).lower(),
        str(gold.get("tool", "")).lower(),
        str(vehicle_state.get("weather", "")).lower(),
        str(vehicle_state.get("gear", "")).lower(),
        "安全",
        "风险",
        "制动",
        "疲劳",
        "safe",
        "risk",
    ]
    keywords = [keyword for keyword in keywords if keyword and keyword != "none"]
    if not keywords:
        return 0.0
    return 1.0 if any(keyword in reason for keyword in keywords) else 0.0


def tool_argument_accuracy(rows: list[dict[str, Any]]) -> float:
    tool_rows = [row for row in rows if row.get("meta", {}).get("task_type") in {"tool_call", "personalized_service", "safety_rejection"}]
    if not tool_rows:
        return 0.0
    correct = 0
    for row in tool_rows:
        pred = parse_model_output(row.get("prediction"))["data"]
        pred_args = pred.get("arguments")
        gold_args = row.get("gold", {}).get("arguments")
        if isinstance(pred_args, dict) and isinstance(gold_args, dict) and all(pred_args.get(k) == v for k, v in gold_args.items()):
            correct += 1
    return correct / len(tool_rows)


def collect_bad_cases(rows: list[dict[str, Any]], reward_threshold: float) -> list[dict[str, Any]]:
    bad_cases = []
    for row in rows:
        reward = total_reward(row.get("prediction"), row.get("gold"), row.get("vehicle_state"), row.get("meta"))
        parsed = parse_model_output(row.get("prediction"))
        reasons = []
        if not parsed["ok"]:
            reasons.append(parsed.get("parse_error") or "invalid_json")
        if reward["total"] < reward_threshold:
            reasons.append("low_reward")
        if schema_completeness(row) < 1.0:
            reasons.append("schema_incomplete")
        pred_task = parsed["data"].get("task") if parsed["ok"] else None
        gold_task = row.get("meta", {}).get("task_type") or row.get("gold", {}).get("task")
        if pred_task and gold_task and pred_task != gold_task:
            reasons.append("task_mismatch")
        if reasons:
            bad_cases.append({**row, "bad_case_reasons": reasons, "reward": reward})
    return bad_cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_table(metrics: dict[str, float]) -> None:
    print("+------------------------+---------+")
    print("| metric                 | value   |")
    print("+------------------------+---------+")
    for key, value in metrics.items():
        print(f"| {key:<22} | {value:<7.4f} |")
    print("+------------------------+---------+")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all DriveMind-VL local eval metrics.")
    parser.add_argument("--predictions", default="outputs/eval_results/base_predictions.jsonl")
    parser.add_argument("--output", default="outputs/eval_results/metrics.json")
    parser.add_argument("--bad_cases_output", default="outputs/cases/bad_cases.jsonl")
    parser.add_argument("--bad_case_reward_threshold", type=float, default=0.65)
    args = parser.parse_args()
    rows = load_jsonl(Path(args.predictions))
    metrics = compute_metrics(rows)
    bad_cases = collect_bad_cases(rows, args.bad_case_reward_threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    write_jsonl(Path(args.bad_cases_output), bad_cases)
    print_table(metrics)
    print(f"wrote metrics to {args.output}")
    print(f"wrote {len(bad_cases)} bad cases to {args.bad_cases_output}")


if __name__ == "__main__":
    main()
