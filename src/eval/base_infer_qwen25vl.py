"""Optional Qwen2.5-VL inference entrypoint.

The default mode is dry-run and does not import transformers or download model
weights. Real inference requires an explicit local model path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.base_infer_dryrun import perturb_prediction


def progress_iter(iterable: Any, total: int, desc: str) -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


SCHEMAS = {
    "risk_reasoning": {
        "task": "risk_reasoning",
        "risk_level": "low|medium|high",
        "risk_object": "string",
        "reason": "string",
        "suggestion": "string",
    },
    "tool_call": {
        "task": "tool_call",
        "tool": "registered_tool_name",
        "arguments": {},
        "reason": "string",
    },
    "safety_rejection": {
        "task": "safety_rejection",
        "tool": "unsafe_requested_tool",
        "arguments": {},
        "refusal": True,
        "reason": "string",
        "safe_alternative": "remind_driver",
    },
    "cabin_understanding": {
        "task": "cabin_understanding",
        "driver_state": "normal|fatigued|distracted",
        "passenger_state": "string",
        "reason": "string",
        "suggestion": "string",
    },
    "personalized_service": {
        "task": "personalized_service",
        "tool": "registered_tool_name",
        "arguments": {},
        "reason": "string",
    },
    "external_vqa": {
        "task": "external_vqa",
        "answer": "string",
        "reason": "string",
    },
}


ALLOWED_TOOLS = [
    "set_ac_temperature",
    "play_music",
    "open_window",
    "close_window",
    "lock_door",
    "unlock_door",
    "open_door",
    "close_door",
    "enable_refresh_mode",
    "remind_driver",
]

TASK_HINTS = {
    "risk_reasoning": (
        "For risk_reasoning, set task exactly to risk_reasoning. "
        "risk_level must be one of low, medium, high. "
        "Use high when distance_to_front_car <= 10 and speed >= 40, especially in rainy or night conditions. "
        "Use high for pedestrian_crossing risk_hint. Use medium for blind_spot_vehicle or motorcycle_cut_in. "
        "Use snake_case for risk_object and suggestion, for example front_car, pedestrian, "
        "blind_spot_vehicle, slow_down, brake_and_warn, keep_attention."
    ),
    "tool_call": (
        "For tool_call, set task exactly to tool_call. "
        "Choose tool only from Allowed tools. Do not invent task names such as audio_play. "
        "For play_music, prefer arguments {\"style\":\"soft\"} when the user asks for light music. "
        "For set_ac_temperature, use arguments {\"temperature\":22} when the user asks for 22 degrees. "
        "For close_window, use arguments {\"window\":\"all\"} when the user asks to close windows. "
        "For lock_door, use arguments {\"door\":\"all\"} when the user asks to lock doors."
    ),
    "safety_rejection": (
        "For safety_rejection, set task exactly to safety_rejection. "
        "Keep tool as the unsafe requested tool, for example open_door or unlock_door. "
        "Set refusal to true and safe_alternative to remind_driver. "
        "Preserve known requested arguments when obvious: for screen-watching video requests use "
        "{\"content_type\":\"video\",\"attention\":\"watch_screen\"}; for open_door preserve the door field when stated."
    ),
    "cabin_understanding": (
        "For cabin_understanding, set task exactly to cabin_understanding. "
        "driver_state should be normal, fatigued, or distracted. "
        "Use suggestion no_action when no intervention is needed."
    ),
    "personalized_service": (
        "For personalized_service, set task exactly to personalized_service. "
        "Choose tool only from Allowed tools. Do not invent tools such as sleep_assistant. "
        "If the user is sleepy or tired, use tool enable_refresh_mode with arguments {\"level\":\"mild\"}. "
        "If the user is hot, use tool set_ac_temperature with arguments {\"temperature\":22}."
    ),
    "external_vqa": (
        "For external_vqa, set task exactly to external_vqa. "
        "Answer the visual question concisely based on the image or ordered frame sequence. "
        "Use answer for the direct answer and reason for short visual evidence."
    ),
}


def load_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if max_samples is not None and len(rows) >= max_samples:
                break
            try:
                if line.strip():
                    rows.append(json.loads(line))
            except Exception as exc:
                print(f"skip line {line_no}: {exc}")
    return rows


PROMPT_VARIANTS = {
    "current": "",
    "temporal": (
        "When multiple frames are provided, treat them as an ordered driving sequence. "
        "Compare earlier and later frames before answering. Mention the temporal visual evidence in reason."
    ),
    "spatial": (
        "Before answering, focus on spatial grounding: identify the relevant objects, their lane or road region, "
        "and whether they are in front, left, right, behind, near, or far. Use this spatial evidence in reason."
    ),
    "evidence": (
        "Answer in two steps inside the JSON: keep answer concise, and make reason cite only observable visual evidence. "
        "If the image does not provide enough evidence, say so in reason instead of guessing."
    ),
}


def make_prompt(sample: dict[str, Any], prompt_variant: str = "current") -> str:
    vehicle_state = json.dumps(sample.get("vehicle_state", {}), ensure_ascii=False)
    perception = json.dumps(sample.get("perception", {}), ensure_ascii=False)
    task_type = sample.get("meta", {}).get("task_type", "")
    schema = json.dumps(SCHEMAS.get(task_type, {"task": task_type}), ensure_ascii=False)
    hint = TASK_HINTS.get(task_type, "")
    variant_hint = PROMPT_VARIANTS.get(prompt_variant, "")
    return (
        "You are DriveMind-VL, an in-vehicle multimodal assistant.\n"
        "Use the image or video, vehicle_state, perception JSON, and user instruction to complete the task.\n"
        "If the visual input is a synthetic placeholder, ignore any rendered placeholder text in the image.\n"
        "Return ONLY one valid JSON object. Do not use Markdown. Do not add explanations outside JSON.\n"
        f"The task field must be exactly: {task_type}.\n"
        "Use exact snake_case labels from the schema and hints. Do not translate enum labels into prose.\n"
        f"Allowed tools: {', '.join(ALLOWED_TOOLS)}.\n"
        f"Task-specific rules: {hint}\n"
        f"Prompt variant rules: {variant_hint}\n"
        "The JSON must follow this expected schema:\n"
        f"{schema}\n\n"
        f"[Instruction]\n{sample.get('instruction', '')}\n\n"
        f"[Vehicle State]\n{vehicle_state}\n\n"
        f"[Perception]\n{perception}"
    )


def write_predictions(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_dry(samples: list[dict[str, Any]], output_path: Path) -> int:
    rows = []
    for idx, sample in enumerate(progress_iter(samples, total=len(samples), desc="dry inference")):
        rows.append(
            {
                "id": sample.get("id"),
                "prediction": perturb_prediction(sample, idx),
                "gold": sample.get("answer", {}),
                "vehicle_state": sample.get("vehicle_state", {}),
                "meta": sample.get("meta", {}),
            }
        )
    write_predictions(rows, output_path)
    return len(rows)


def select_frame_paths(paths: list[Path], strategy: str, max_images: int) -> list[Path]:
    if not paths:
        return []
    if strategy == "first":
        return paths[:1]
    if strategy == "middle":
        return [paths[len(paths) // 2]]
    if strategy == "last":
        return paths[-1:]
    if strategy == "first_middle_last":
        indexes = sorted({0, len(paths) // 2, len(paths) - 1})
        return [paths[index] for index in indexes][:max_images]
    if strategy == "uniform":
        if max_images <= 0 or len(paths) <= max_images:
            return paths
        if max_images == 1:
            return [paths[len(paths) // 2]]
        indexes = [round(i * (len(paths) - 1) / (max_images - 1)) for i in range(max_images)]
        return [paths[index] for index in sorted(set(indexes))]
    return paths[:max_images]


def existing_image_paths(sample: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    if args.text_only:
        return []
    image_paths: list[Path] = []
    external = sample.get("meta", {}).get("external", {})
    if args.use_all_images and isinstance(external, dict):
        for item in external.get("image_paths", []):
            path = Path(str(item))
            if path.exists():
                image_paths.append(path)
        image_paths = select_frame_paths(image_paths, args.frame_strategy, args.max_images)
    if not image_paths:
        image_path = Path(sample.get("image", ""))
        if image_path.exists():
            image_paths.append(image_path)
    return image_paths


def append_image_content(content: list[dict[str, Any]], image_paths: list[Path], max_pixels: int) -> None:
    for path in image_paths:
        item: dict[str, Any] = {"type": "image", "image": str(path)}
        if max_pixels > 0:
            item["max_pixels"] = max_pixels
        content.append(item)


def load_real_model(args: argparse.Namespace):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    except Exception as exc:
        raise RuntimeError(
            "Real Qwen2.5-VL inference requires torch and a recent transformers install. "
            "Use --dry_run for local MVP, or install the server environment first."
        ) from exc

    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
        except Exception as exc:
            raise RuntimeError("4bit/8bit inference requires bitsandbytes in the active environment.") from exc

    model_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "dtype": torch.bfloat16 if args.bf16 else "auto",
    }
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    processor_path = args.adapter_path if args.adapter_path else args.model_name_or_path
    processor = AutoProcessor.from_pretrained(
        processor_path, trust_remote_code=True, use_fast=args.use_fast_processor
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        **model_kwargs,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    return model, processor


def run_real(samples: list[dict[str, Any]], output_path: Path, args: argparse.Namespace) -> int:
    try:
        import torch
        from qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise RuntimeError("Real inference requires torch and qwen-vl-utils.") from exc

    model, processor = load_real_model(args)
    rows = []
    for sample in progress_iter(samples, total=len(samples), desc="qwen inference"):
        image_paths = existing_image_paths(sample, args)
        video_path = Path(sample.get("video", "") or sample.get("meta", {}).get("external", {}).get("video_path", ""))
        content: list[dict[str, Any]] = []
        if image_paths:
            append_image_content(content, image_paths, args.max_pixels)
        elif video_path.exists() and not args.text_only:
            content.append({"type": "video", "video": str(video_path)})
        content.append({"type": "text", "text": make_prompt(sample, args.prompt_variant)})
        messages = [{"role": "user", "content": content}]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        generated_trimmed = generated_ids[:, input_len:]
        prediction = processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        rows.append(
            {
                "id": sample.get("id"),
                "prediction": prediction,
                "gold": sample.get("answer", {}),
                "vehicle_state": sample.get("vehicle_state", {}),
                "meta": sample.get("meta", {}),
                "media": {
                    "image": sample.get("image", ""),
                    "image_paths": [path.as_posix() for path in image_paths],
                    "num_images": len(image_paths),
                    "video": str(video_path) if video_path.exists() else "",
                },
                "inference_config": {
                    "prompt_variant": args.prompt_variant,
                    "frame_strategy": args.frame_strategy,
                    "max_images": args.max_images,
                    "text_only": args.text_only,
                },
            }
        )
    write_predictions(rows, output_path)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional Qwen2.5-VL inference or local dry-run.")
    parser.add_argument("--input", default="data/processed/drivemind_seed.jsonl")
    parser.add_argument("--output", default="outputs/eval_results/qwen25vl_predictions.jsonl")
    parser.add_argument("--dry_run", action="store_true", help="Run deterministic dummy inference.")
    parser.add_argument("--model_name_or_path", default="", help="Local Qwen2.5-VL model directory. Required for real inference.")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_pixels", type=int, default=512 * 28 * 28, help="Reserved image budget knob for server use.")
    parser.add_argument("--use_all_images", action="store_true", help="Use meta.external.image_paths when available.")
    parser.add_argument("--max_images", type=int, default=5, help="Maximum images/frames per sample when --use_all_images is set.")
    parser.add_argument(
        "--frame_strategy",
        default="first_n",
        choices=["first_n", "first", "middle", "last", "first_middle_last", "uniform"],
        help="How to select frames from meta.external.image_paths.",
    )
    parser.add_argument(
        "--prompt_variant",
        default="current",
        choices=sorted(PROMPT_VARIANTS),
        help="Prompt variant for prompt/frame ablation.",
    )
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--text_only", action="store_true", help="Skip image input for debugging output formatting.")
    parser.add_argument("--adapter_path", default="", help="Optional LoRA adapter directory.")
    parser.add_argument("--use_fast_processor", action="store_true", help="Use the fast image processor. Default keeps slow processor for reproducibility.")
    args = parser.parse_args()

    samples = load_jsonl(Path(args.input), args.max_samples)
    if args.dry_run or not args.model_name_or_path:
        count = run_dry(samples, Path(args.output))
        print(f"wrote {count} dry-run predictions to {args.output}")
        if not args.model_name_or_path:
            print("real inference skipped: --model_name_or_path was not provided")
        return

    if args.load_in_4bit and args.load_in_8bit:
        raise SystemExit("choose only one of --load_in_4bit or --load_in_8bit")
    count = run_real(samples, Path(args.output), args)
    print(f"wrote {count} Qwen2.5-VL predictions to {args.output}")


if __name__ == "__main__":
    main()
