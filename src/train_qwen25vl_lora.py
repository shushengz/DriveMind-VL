"""Minimal Qwen2.5-VL LoRA SFT smoke training.

This script is intentionally small and explicit. It is for the first 3B LoRA
smoke experiment, not for full training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.base_infer_qwen25vl import make_prompt


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


def build_messages(sample: dict[str, Any], include_assistant: bool) -> list[dict[str, Any]]:
    image_path = sample.get("image", "")
    content: list[dict[str, Any]] = []
    if image_path and Path(image_path).exists():
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": make_prompt(sample)})
    messages = [{"role": "user", "content": content}]
    if include_assistant:
        answer = json.dumps(sample.get("answer", {}), ensure_ascii=False, separators=(",", ":"))
        messages.append({"role": "assistant", "content": answer})
    return messages


def encode_sample(processor: Any, sample: dict[str, Any], device: str) -> dict[str, Any]:
    import torch
    from qwen_vl_utils import process_vision_info

    prompt_messages = build_messages(sample, include_assistant=False)
    full_messages = build_messages(sample, include_assistant=True)

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

    image_inputs, video_inputs = process_vision_info(full_messages)
    prompt_inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_inputs = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    labels = full_inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs["labels"] = labels
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in full_inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal Qwen2.5-VL LoRA SFT smoke training.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", default="data/processed/drivemind_train.jsonl")
    parser.add_argument("--output_dir", default="outputs/checkpoints/qwen25vl_3b_lora_smoke")
    parser.add_argument("--max_samples", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(Path(args.train_file), args.max_samples)
    print(f"loaded train samples: {len(samples)}")
    if args.dry_run:
        print("dry_run: skip model loading and training")
        return

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    total_steps = args.epochs * math.ceil(len(samples) / args.gradient_accumulation_steps)
    print(f"epochs={args.epochs} optimizer_steps~={total_steps}")

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for idx, sample in enumerate(samples):
            batch = encode_sample(processor, sample, device)
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            if (idx + 1) % args.gradient_accumulation_steps == 0 or idx + 1 == len(samples):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                print(f"epoch={epoch + 1} step={global_step} loss={loss.item() * args.gradient_accumulation_steps:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()

