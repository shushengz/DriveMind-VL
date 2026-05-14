"""Dummy planner for the local MVP."""

from __future__ import annotations

from typing import Any


def make_dummy_plan(instruction: str, vehicle_state: dict[str, Any], perception: dict[str, Any]) -> dict[str, Any]:
    text = instruction.lower()
    if "开门" in instruction or "打开车门" in instruction or "车门" in instruction or "open door" in text:
        return {"task": "tool_call", "tool": "open_door", "arguments": {"door": "left_front"}, "reason": "用户请求打开车门"}
    if "空调" in instruction or "temperature" in text:
        return {"task": "tool_call", "tool": "set_ac_temperature", "arguments": {"temperature": 22}, "reason": "调节座舱温度"}
    if "困" in instruction or "sleepy" in text:
        return {"task": "personalized_service", "tool": "enable_refresh_mode", "arguments": {"level": "mild"}, "reason": "用户可能疲劳"}
    if perception.get("risk_hint"):
        return {"task": "risk_reasoning", "risk_level": "high", "risk_object": "front_car", "reason": "检测到前方风险目标", "suggestion": "slow_down"}
    return {"task": "risk_reasoning", "risk_level": "low", "risk_object": "none", "reason": "未发现明显风险", "suggestion": "keep_attention"}
