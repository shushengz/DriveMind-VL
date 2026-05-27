"""Prepared Qwen3-VL strict visual-control evaluator.

This shares the proven setting alignment/scoring path with the Qwen2.5-VL
evaluator, replacing only the model loader. Stage 15 does not execute it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval import run_visual_control_eval as base


def load_qwen3_model(args: Any):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": args.device_map,
        "torch_dtype": torch.bfloat16 if args.bf16 else "auto",
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=args.use_fast_processor)
    model = AutoModelForImageTextToText.from_pretrained(args.model_path, **kwargs)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, processor


def main() -> None:
    base.load_model = load_qwen3_model
    base.main()


if __name__ == "__main__":
    main()
