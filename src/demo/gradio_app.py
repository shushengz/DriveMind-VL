"""Gradio demo skeleton for DriveMind-VL local MVP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.agent.planner import make_dummy_plan
    from src.agent.safety_guard import check_safety
    from src.perception.build_perception_json import build_perception
    from src.rewards.total_reward import total_reward
except Exception:
    from agent.planner import make_dummy_plan
    from agent.safety_guard import check_safety
    from perception.build_perception_json import build_perception
    from rewards.total_reward import total_reward


EXAMPLES = [
    ["data/images/road_scene_0.jpg", '{"speed":35,"gear":"D","weather":"clear","distance_to_front_car":20,"time":"day"}', "现在帮我打开车门。"],
    ["data/images/cabin_scene_0.jpg", '{"speed":0,"gear":"P","weather":"clear","distance_to_front_car":30,"time":"day"}', "请把空调调到22度。"],
    ["data/images/road_scene_1.jpg", '{"speed":45,"gear":"D","weather":"rainy","distance_to_front_car":8.5,"time":"night"}', "请判断当前驾驶风险等级，并输出JSON。"],
    ["data/images/cabin_scene_1.jpg", '{"speed":50,"gear":"D","weather":"clear","distance_to_front_car":35,"time":"night"}', "我有点困，帮我处理一下。"],
]


def parse_vehicle_state(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def run_dummy(image: Any, vehicle_state_text: str, instruction: str) -> tuple[str, str, str, str]:
    vehicle_state = parse_vehicle_state(vehicle_state_text)
    image_path = image if isinstance(image, str) else ""
    perception = build_perception(image_path)
    model_output = make_dummy_plan(instruction, vehicle_state, perception)
    guard = check_safety(model_output, vehicle_state)
    reward = total_reward(model_output, model_output, vehicle_state, {"task_type": model_output.get("task")})
    return (
        json.dumps(perception, ensure_ascii=False, indent=2),
        json.dumps(model_output, ensure_ascii=False, indent=2),
        json.dumps(guard, ensure_ascii=False, indent=2),
        json.dumps(reward, ensure_ascii=False, indent=2),
    )


def build_app():
    import gradio as gr

    with gr.Blocks(title="DriveMind-VL Local MVP") as demo:
        gr.Markdown("# DriveMind-VL Local MVP")
        with gr.Row():
            image = gr.Image(type="filepath", label="Image")
            with gr.Column():
                vehicle_state = gr.Textbox(label="Vehicle State JSON", lines=6, value='{"speed":35,"gear":"D","weather":"clear","distance_to_front_car":20,"time":"day"}')
                instruction = gr.Textbox(label="Instruction", lines=3, value="现在帮我打开车门。")
                run_btn = gr.Button("Run")
        with gr.Row():
            perception = gr.Code(label="Perception JSON", language="json")
            model_output = gr.Code(label="Dummy Model Output JSON", language="json")
        with gr.Row():
            safety = gr.Code(label="Safety Guard Result", language="json")
            reward = gr.Code(label="Reward Detail", language="json")
        gr.Examples(examples=EXAMPLES, inputs=[image, vehicle_state, instruction])
        run_btn.click(run_dummy, inputs=[image, vehicle_state, instruction], outputs=[perception, model_output, safety, reward])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriveMind-VL Gradio demo.")
    parser.add_argument("--server_name", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Run one demo call and exit.")
    args = parser.parse_args()
    if args.smoke_test:
        outputs = run_dummy(None, EXAMPLES[0][1], EXAMPLES[0][2])
        print("\n---\n".join(outputs))
        return
    existing_no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    local_hosts = ["localhost", "127.0.0.1"]
    merged_no_proxy = ",".join(dict.fromkeys([item for item in existing_no_proxy.split(",") if item] + local_hosts))
    os.environ["NO_PROXY"] = merged_no_proxy
    os.environ["no_proxy"] = merged_no_proxy
    app = build_app()
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
