"""Build synthetic DriveMind-Instruct seed data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASK_TYPES = [
    "risk_reasoning",
    "tool_call",
    "safety_rejection",
    "cabin_understanding",
    "personalized_service",
]


def ensure_placeholder_images(image_dir: Path) -> list[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "road_scene_0.jpg",
        "road_scene_1.jpg",
        "road_scene_2.jpg",
        "cabin_scene_0.jpg",
        "cabin_scene_1.jpg",
    ]
    paths = [image_dir / name for name in names]
    if all(path.exists() for path in paths):
        return paths

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError("Pillow is required to generate placeholder images. Run scripts/01_install_env.sh.") from exc

    palettes = [
        ((38, 54, 66), (220, 226, 230)),
        ((56, 79, 63), (230, 235, 220)),
        ((76, 65, 54), (238, 228, 218)),
        ((48, 48, 58), (232, 230, 245)),
        ((54, 43, 62), (240, 230, 245)),
    ]
    for idx, path in enumerate(paths):
        if path.exists():
            continue
        bg, fg = palettes[idx]
        img = Image.new("RGB", (640, 360), bg)
        draw = ImageDraw.Draw(img)
        label = path.stem.replace("_", " ")
        draw.rectangle([40, 250, 600, 310], outline=fg, width=3)
        draw.line([80, 360, 300, 190], fill=fg, width=5)
        draw.line([560, 360, 340, 190], fill=fg, width=5)
        draw.text((60, 40), label, fill=fg, font=ImageFont.load_default())
        draw.text((60, 70), "DriveMind-VL placeholder", fill=fg, font=ImageFont.load_default())
        img.save(path, quality=92)
    return paths


def risk_sample(idx: int, image: str) -> dict[str, Any]:
    scenarios = [
        {
            "risk_level": "high",
            "risk_object": "front_car",
            "suggestion": "slow_down",
            "weather": "rainy",
            "time": "night",
            "speed": 45,
            "distance": 8.5,
            "scene": "urban_road",
            "risk_hint": "front_car_close",
            "objects": [{"class": "car", "bbox": [320, 210, 480, 360], "position": "front", "relative_depth": 8.5, "risk_score": 0.83}],
            "reason": "雨天且前车距离较近，当前车速较高，制动风险增加",
        },
        {
            "risk_level": "high",
            "risk_object": "pedestrian",
            "suggestion": "brake_and_warn",
            "weather": "clear",
            "time": "night",
            "speed": 38,
            "distance": 12.0,
            "scene": "crosswalk",
            "risk_hint": "pedestrian_crossing",
            "objects": [{"class": "pedestrian", "bbox": [260, 150, 310, 330], "position": "front_right", "relative_depth": 12.0, "risk_score": 0.91}],
            "reason": "夜间斑马线附近出现行人，车辆仍在行驶，需要及时减速制动",
        },
        {
            "risk_level": "medium",
            "risk_object": "blind_spot_vehicle",
            "suggestion": "keep_lane",
            "weather": "clear",
            "time": "day",
            "speed": 62,
            "distance": 18.0,
            "scene": "highway",
            "risk_hint": "blind_spot_vehicle",
            "objects": [{"class": "car", "bbox": [35, 190, 180, 330], "position": "left_rear", "relative_depth": 18.0, "risk_score": 0.66}],
            "reason": "左后方盲区有车辆，当前车速较高，变道风险增加",
        },
        {
            "risk_level": "medium",
            "risk_object": "motorcycle",
            "suggestion": "increase_distance",
            "weather": "rainy",
            "time": "day",
            "speed": 32,
            "distance": 14.0,
            "scene": "urban_road",
            "risk_hint": "motorcycle_cut_in",
            "objects": [{"class": "motorcycle", "bbox": [390, 190, 470, 345], "position": "front_right", "relative_depth": 14.0, "risk_score": 0.72}],
            "reason": "雨天右前方摩托车可能并线，应保持安全距离",
        },
        {
            "risk_level": "low",
            "risk_object": "none",
            "suggestion": "keep_attention",
            "weather": "clear",
            "time": "day",
            "speed": 28,
            "distance": 32.0,
            "scene": "urban_road",
            "risk_hint": "normal_following",
            "objects": [{"class": "car", "bbox": [330, 220, 450, 345], "position": "front", "relative_depth": 32.0, "risk_score": 0.22}],
            "reason": "当前车距和车速处于可控范围，未发现明显高风险目标",
        },
    ]
    scenario = scenarios[idx % len(scenarios)]
    return {
        "id": f"risk_{idx:06d}",
        "image": image,
        "vehicle_state": {
            "speed": scenario["speed"],
            "weather": scenario["weather"],
            "distance_to_front_car": scenario["distance"],
            "yaw_rate": 0.03,
            "gear": "D",
            "time": scenario["time"],
        },
        "perception": {
            "objects": scenario["objects"],
            "scene": scenario["scene"],
            "risk_hint": scenario["risk_hint"],
        },
        "instruction": "请判断当前驾驶风险等级，并输出JSON。",
        "answer": {
            "task": "risk_reasoning",
            "risk_level": scenario["risk_level"],
            "risk_object": scenario["risk_object"],
            "reason": scenario["reason"],
            "suggestion": scenario["suggestion"],
        },
        "meta": {"task_type": "risk_reasoning", "source": "synthetic_seed", "difficulty": "easy"},
    }


def tool_sample(idx: int, image: str) -> dict[str, Any]:
    tools = [
        ("set_ac_temperature", {"temperature": 22}, "请把空调调到22度。"),
        ("play_music", {"style": "soft"}, "播放轻音乐。"),
        ("close_window", {"window": "all"}, "帮我关上车窗。"),
        ("lock_door", {"door": "all"}, "锁上车门。"),
    ]
    tool, arguments, instruction = tools[idx % len(tools)]
    return {
        "id": f"tool_{idx:06d}",
        "image": image,
        "vehicle_state": {"speed": 0, "weather": "clear", "distance_to_front_car": 30, "yaw_rate": 0.0, "gear": "P", "time": "day"},
        "perception": {"objects": [], "scene": "parking_lot", "risk_hint": "none"},
        "instruction": instruction,
        "answer": {"task": "tool_call", "tool": tool, "arguments": arguments, "reason": "用户请求属于安全座舱控制"},
        "meta": {"task_type": "tool_call", "source": "synthetic_seed", "difficulty": "easy"},
    }


def safety_sample(idx: int, image: str) -> dict[str, Any]:
    cases = [
        {
            "instruction": "现在帮我打开车门。",
            "tool": "open_door",
            "arguments": {"door": "left_front"},
            "vehicle_state": {"speed": 35, "weather": "clear", "distance_to_front_car": 20, "yaw_rate": 0.02, "gear": "D", "time": "day"},
            "reason": "车辆行驶中打开车门存在严重安全风险",
            "difficulty": "medium",
        },
        {
            "instruction": "帮我解锁车门。",
            "tool": "unlock_door",
            "arguments": {"door": "all"},
            "vehicle_state": {"speed": 12, "weather": "clear", "distance_to_front_car": 20, "yaw_rate": 0.02, "gear": "D", "time": "day"},
            "reason": "车辆行驶中解锁车门存在误开门风险",
            "difficulty": "medium",
        },
        {
            "instruction": "我停下了但还在 D 挡，帮我打开车门。",
            "tool": "open_door",
            "arguments": {"door": "right_rear"},
            "vehicle_state": {"speed": 0, "weather": "clear", "distance_to_front_car": 12, "yaw_rate": 0.0, "gear": "D", "time": "day"},
            "reason": "车辆未挂入 P 挡时不能打开车门",
            "difficulty": "hard",
        },
        {
            "instruction": "现在给我播放一段需要盯着中控屏看的视频。",
            "tool": "play_music",
            "arguments": {"content_type": "video", "attention": "watch_screen"},
            "vehicle_state": {"speed": 55, "weather": "clear", "distance_to_front_car": 28, "yaw_rate": 0.01, "gear": "D", "time": "night"},
            "reason": "高速行驶时要求驾驶员观看屏幕会分散注意力",
            "difficulty": "hard",
        },
    ]
    case = cases[idx % len(cases)]
    return {
        "id": f"safety_{idx:06d}",
        "image": image,
        "vehicle_state": case["vehicle_state"],
        "perception": {"objects": [], "scene": "urban_road", "risk_hint": "moving_vehicle"},
        "instruction": case["instruction"],
        "answer": {
            "task": "safety_rejection",
            "tool": case["tool"],
            "arguments": case["arguments"],
            "refusal": True,
            "reason": case["reason"],
            "safe_alternative": "remind_driver",
        },
        "meta": {"task_type": "safety_rejection", "source": "synthetic_seed", "difficulty": case["difficulty"]},
    }


def cabin_sample(idx: int, image: str) -> dict[str, Any]:
    sleepy = idx % 2 == 0
    return {
        "id": f"cabin_{idx:06d}",
        "image": image,
        "vehicle_state": {"speed": 42, "weather": "clear", "distance_to_front_car": 25, "yaw_rate": 0.01, "gear": "D", "time": "night"},
        "perception": {
            "objects": [{"class": "driver_face", "bbox": [250, 90, 390, 260], "position": "front_left", "relative_depth": 1.0, "risk_score": 0.65 if sleepy else 0.2}],
            "scene": "cabin",
            "risk_hint": "driver_fatigue" if sleepy else "normal_cabin",
        },
        "instruction": "请理解当前舱内状态，并输出JSON。",
        "answer": {
            "task": "cabin_understanding",
            "driver_state": "fatigued" if sleepy else "normal",
            "passenger_state": "normal",
            "reason": "驾驶员眼部状态显示可能疲劳" if sleepy else "驾驶员姿态和注意力正常",
            "suggestion": "remind_driver" if sleepy else "no_action",
        },
        "meta": {"task_type": "cabin_understanding", "source": "synthetic_seed", "difficulty": "easy"},
    }


def service_sample(idx: int, image: str) -> dict[str, Any]:
    sleepy = idx % 2 == 0
    return {
        "id": f"service_{idx:06d}",
        "image": image,
        "vehicle_state": {"speed": 50 if sleepy else 0, "weather": "clear", "distance_to_front_car": 35, "yaw_rate": 0.01, "gear": "D" if sleepy else "P", "time": "night" if sleepy else "day"},
        "perception": {"objects": [], "scene": "cabin", "risk_hint": "driver_fatigue" if sleepy else "comfort_request"},
        "instruction": "我有点困，帮我处理一下。" if sleepy else "我觉得有点热，帮我舒服一点。",
        "answer": {
            "task": "personalized_service",
            "tool": "enable_refresh_mode" if sleepy else "set_ac_temperature",
            "arguments": {"level": "mild"} if sleepy else {"temperature": 22},
            "reason": "用户表达疲劳，应启用轻度提神并提醒专注驾驶" if sleepy else "用户表达温度不适，停车状态可调节空调",
        },
        "meta": {"task_type": "personalized_service", "source": "synthetic_seed", "difficulty": "easy"},
    }


BUILDERS = {
    "risk_reasoning": risk_sample,
    "tool_call": tool_sample,
    "safety_rejection": safety_sample,
    "cabin_understanding": cabin_sample,
    "personalized_service": service_sample,
}


def build_samples(output: Path, image_dir: Path, num_samples: int) -> list[dict[str, Any]]:
    images = ensure_placeholder_images(image_dir)
    road_images = [str(path.as_posix()) for path in images if "road" in path.name]
    cabin_images = [str(path.as_posix()) for path in images if "cabin" in path.name]

    samples = []
    for idx in range(num_samples):
        task = TASK_TYPES[idx % len(TASK_TYPES)]
        image_pool = cabin_images if task in {"cabin_understanding", "personalized_service"} else road_images
        image = image_pool[idx % len(image_pool)]
        samples.append(BUILDERS[task](idx, image))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic DriveMind-Instruct seed data.")
    parser.add_argument("--output", default="data/processed/drivemind_seed.jsonl")
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    samples = build_samples(Path(args.output), Path(args.image_dir), args.num_samples)
    counts = Counter(sample["meta"]["task_type"] for sample in samples)
    print(f"wrote {len(samples)} samples to {args.output}")
    for task in TASK_TYPES:
        print(f"{task}: {counts[task]}")


if __name__ == "__main__":
    main()
