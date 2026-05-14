"""Mock vehicle tools for the local MVP.

These functions do not control real hardware. They only return structured
execution records so the agent loop, safety guard, rewards, and demo can run.
"""

from __future__ import annotations

from typing import Any, Callable


def _result(tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"tool": tool, "status": "success", "arguments": arguments or {}}


def set_ac_temperature(**kwargs: Any) -> dict[str, Any]:
    return _result("set_ac_temperature", kwargs)


def play_music(**kwargs: Any) -> dict[str, Any]:
    return _result("play_music", kwargs)


def open_window(**kwargs: Any) -> dict[str, Any]:
    return _result("open_window", kwargs)


def close_window(**kwargs: Any) -> dict[str, Any]:
    return _result("close_window", kwargs)


def lock_door(**kwargs: Any) -> dict[str, Any]:
    return _result("lock_door", kwargs)


def unlock_door(**kwargs: Any) -> dict[str, Any]:
    return _result("unlock_door", kwargs)


def open_door(**kwargs: Any) -> dict[str, Any]:
    return _result("open_door", kwargs)


def close_door(**kwargs: Any) -> dict[str, Any]:
    return _result("close_door", kwargs)


def enable_refresh_mode(**kwargs: Any) -> dict[str, Any]:
    return _result("enable_refresh_mode", kwargs)


def remind_driver(**kwargs: Any) -> dict[str, Any]:
    return _result("remind_driver", kwargs)


TOOL_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "set_ac_temperature": set_ac_temperature,
    "play_music": play_music,
    "open_window": open_window,
    "close_window": close_window,
    "lock_door": lock_door,
    "unlock_door": unlock_door,
    "open_door": open_door,
    "close_door": close_door,
    "enable_refresh_mode": enable_refresh_mode,
    "remind_driver": remind_driver,
}


def get_registered_tools() -> set[str]:
    return set(TOOL_REGISTRY)


def run_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return {"tool": tool_name, "status": "error", "error": "unknown_tool", "arguments": arguments or {}}
    try:
        return tool(**(arguments or {}))
    except Exception as exc:
        return {"tool": tool_name, "status": "error", "error": str(exc), "arguments": arguments or {}}

