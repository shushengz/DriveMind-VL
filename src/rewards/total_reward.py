"""Aggregate reward for DriveMind-VL local MVP."""

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
    from src.rewards.format_reward import format_reward
    from src.rewards.risk_reward import risk_reward
    from src.rewards.tool_reward import tool_reward
    from src.rewards.safety_reward import safety_reward
except Exception:
    from agent.output_parser import parse_model_output
    from rewards.format_reward import format_reward
    from rewards.risk_reward import risk_reward
    from rewards.tool_reward import tool_reward
    from rewards.safety_reward import safety_reward


def reasoning_reward(pred: Any, gold: dict[str, Any] | None = None, vehicle_state: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> float:
    try:
        pred_obj = parse_model_output(pred)["data"]
        reason = str(pred_obj.get("reason", "")).lower()
        gold = gold or {}
        vehicle_state = vehicle_state or {}
        score = 0.0
        objects = [str(gold.get("risk_object", "")).lower(), str(gold.get("tool", "")).lower()]
        if any(obj and obj != "none" and obj in reason for obj in objects):
            score += 0.35
        if any(str(vehicle_state.get(k, "")).lower() in reason for k in ("weather", "gear", "time") if vehicle_state.get(k)):
            score += 0.25
        safety_terms = ("安全", "风险", "制动", "疲劳", "safe", "risk")
        if any(term in reason for term in safety_terms):
            score += 0.4
        return min(score, 1.0)
    except Exception:
        return 0.0


def task_reward(pred: Any, gold: dict[str, Any] | None = None, vehicle_state: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> float:
    task = (meta or {}).get("task_type") or (gold or {}).get("task")
    if task == "risk_reasoning":
        return risk_reward(pred, gold, vehicle_state, meta)
    if task in {"tool_call", "personalized_service"}:
        return tool_reward(pred, gold, vehicle_state, meta)
    if task == "safety_rejection":
        return max(safety_reward(pred, gold, vehicle_state, meta), 0.0)
    try:
        pred_obj = parse_model_output(pred)["data"]
        return 1.0 if pred_obj.get("task") == task else 0.0
    except Exception:
        return 0.0


def total_reward(pred: Any, gold: dict[str, Any] | None = None, vehicle_state: dict[str, Any] | None = None, meta: dict[str, Any] | None = None) -> dict[str, float]:
    try:
        fmt = format_reward(pred, gold, vehicle_state, meta)
        task = task_reward(pred, gold, vehicle_state, meta)
        safety = safety_reward(pred, gold, vehicle_state, meta)
        reasoning = reasoning_reward(pred, gold, vehicle_state, meta)
        total = 0.25 * fmt + 0.30 * task + 0.25 * safety + 0.20 * reasoning
        return {
            "format": round(fmt, 4),
            "task": round(task, 4),
            "safety": round(safety, 4),
            "reasoning": round(reasoning, 4),
            "total": round(total, 4),
        }
    except Exception:
        return {"format": 0.0, "task": 0.0, "safety": 0.0, "reasoning": 0.0, "total": 0.0}


def demo() -> dict[str, Any]:
    gold = {
        "task": "risk_reasoning",
        "risk_level": "high",
        "risk_object": "front_car",
        "reason": "雨天且前车距离较近，制动风险增加",
        "suggestion": "slow_down",
    }
    pred = {
        "task": "risk_reasoning",
        "risk_level": "high",
        "risk_object": "front_car",
        "reason": "雨天 front_car 距离近，存在制动风险",
        "suggestion": "slow_down",
    }
    vehicle_state = {"speed": 45, "weather": "rainy", "gear": "D", "time": "night"}
    meta = {"task_type": "risk_reasoning"}
    return {"pred": pred, "gold": gold, "reward": total_reward(pred, gold, vehicle_state, meta)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute DriveMind-VL aggregate reward.")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--pred", default="")
    parser.add_argument("--gold", default="{}")
    parser.add_argument("--vehicle_state", default="{}")
    parser.add_argument("--meta", default="{}")
    args = parser.parse_args()
    if args.demo:
        print(json.dumps(demo(), ensure_ascii=False, indent=2))
        return
    try:
        pred: Any = json.loads(args.pred) if args.pred else {}
        gold = json.loads(args.gold)
        vehicle_state = json.loads(args.vehicle_state)
        meta = json.loads(args.meta)
    except Exception as exc:
        print(json.dumps({"error": f"invalid json input: {exc}"}, ensure_ascii=False))
        return
    print(json.dumps(total_reward(pred, gold, vehicle_state, meta), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
