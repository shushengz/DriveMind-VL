"""Rule-based safety guard for vehicle tool calls."""

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
    from src.agent.tools import get_registered_tools
except Exception:
    from tools import get_registered_tools


SCREEN_WATCH_KEYWORDS = ("watch_screen", "view_screen", "观看屏幕", "看屏幕", "盯着屏幕", "video")


def _extract_tool(model_output_json: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    tool = model_output_json.get("tool") or model_output_json.get("action")
    arguments = model_output_json.get("arguments") or model_output_json.get("args") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return tool, arguments


def check_safety(model_output_json: dict[str, Any], vehicle_state: dict[str, Any]) -> dict[str, Any]:
    try:
        tool, arguments = _extract_tool(model_output_json or {})
        task = model_output_json.get("task")
        speed = float(vehicle_state.get("speed", 0) or 0)
        gear = str(vehicle_state.get("gear", "")).upper()
        registered = get_registered_tools()

        if not tool and task not in {"tool_call", "personalized_service", "safety_rejection"}:
            return {
                "final_decision": "allowed",
                "blocked_reason": "",
                "safe_action": {"tool": "no_action", "arguments": {}},
            }

        if not tool or tool not in registered:
            return {
                "final_decision": "blocked",
                "blocked_reason": "unknown_or_missing_tool",
                "safe_action": {"tool": "remind_driver", "arguments": {"message": "无法执行未知车控动作。"}},
            }

        if tool == "open_door" and speed > 0:
            return {
                "final_decision": "blocked",
                "blocked_reason": "speed_positive_open_door_forbidden",
                "safe_action": {"tool": "remind_driver", "arguments": {"message": "车辆行驶中不能打开车门。"}},
            }
        if tool == "unlock_door" and speed > 0:
            return {
                "final_decision": "blocked",
                "blocked_reason": "speed_positive_unlock_door_forbidden",
                "safe_action": {"tool": "remind_driver", "arguments": {"message": "车辆行驶中不能解锁车门。"}},
            }
        if tool == "open_door" and gear != "P":
            return {
                "final_decision": "blocked",
                "blocked_reason": "non_park_open_door_forbidden",
                "safe_action": {"tool": "remind_driver", "arguments": {"message": "请先挂入 P 挡再打开车门。"}},
            }

        joined = json.dumps(model_output_json, ensure_ascii=False).lower()
        if speed > 30 and any(keyword.lower() in joined for keyword in SCREEN_WATCH_KEYWORDS):
            return {
                "final_decision": "blocked",
                "blocked_reason": "screen_attention_action_forbidden_above_30",
                "safe_action": {"tool": "remind_driver", "arguments": {"message": "高速行驶时请保持注视道路。"}},
            }

        return {
            "final_decision": "allowed",
            "blocked_reason": "",
            "safe_action": {"tool": tool, "arguments": arguments},
        }
    except Exception as exc:
        return {
            "final_decision": "blocked",
            "blocked_reason": f"safety_guard_error: {exc}",
            "safe_action": {"tool": "remind_driver", "arguments": {"message": "安全校验失败，已拒绝执行。"}},
        }


def run_demo() -> list[dict[str, Any]]:
    cases = [
        {
            "name": "speed=35 open_door blocked",
            "model_output_json": {"task": "tool_call", "tool": "open_door", "arguments": {"door": "left_front"}},
            "vehicle_state": {"speed": 35, "gear": "D"},
        },
        {
            "name": "speed=0 gear=P open_door allowed",
            "model_output_json": {"task": "tool_call", "tool": "open_door", "arguments": {"door": "right_rear"}},
            "vehicle_state": {"speed": 0, "gear": "P"},
        },
        {
            "name": "speed=0 set_ac_temperature allowed",
            "model_output_json": {"task": "tool_call", "tool": "set_ac_temperature", "arguments": {"temperature": 22}},
            "vehicle_state": {"speed": 0, "gear": "P"},
        },
    ]
    results = []
    for case in cases:
        result = check_safety(case["model_output_json"], case["vehicle_state"])
        results.append({"case": case["name"], "result": result})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriveMind-VL safety guard.")
    parser.add_argument("--demo", action="store_true", help="Run built-in safety guard demo.")
    parser.add_argument("--model_output_json", default="", help="JSON string from model output.")
    parser.add_argument("--vehicle_state", default="{}", help="Vehicle state JSON string.")
    args = parser.parse_args()

    if args.demo:
        print(json.dumps(run_demo(), ensure_ascii=False, indent=2))
        return

    try:
        model_output = json.loads(args.model_output_json)
        vehicle_state = json.loads(args.vehicle_state)
    except Exception as exc:
        print(json.dumps({"error": f"invalid input json: {exc}"}, ensure_ascii=False))
        return
    print(json.dumps(check_safety(model_output, vehicle_state), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
