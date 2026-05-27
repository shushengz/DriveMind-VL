"""Train a small Qwen2.5-VL DPO-v8 LoRA checkpoint from the r3 adapter.

This smoke trainer deliberately uses a frozen r3 reference and never falls
back to a base-model reference. Default ``--dry_run`` validates artifacts and
writes run metadata without loading a model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shlex
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            for side in ("chosen", "rejected"):
                if not isinstance(row.get(side), dict) or set(row[side]) != {"answer"}:
                    raise ValueError(f"{path}:{line_no}: {side} must be answer-only JSON")
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"no preference rows found: {path}")
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "yes"}:
        return True
    if value.lower() in {"false", "no"}:
        return False
    try:
        return float(value) if any(ch in value for ch in ".eE") else int(value)
    except ValueError:
        return value.strip("'\"")


def load_simple_yaml(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if not path.exists():
        raise FileNotFoundError(f"missing DPO config: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or ":" not in clean:
            continue
        key, value = clean.split(":", 1)
        values[key.strip()] = parse_scalar(value)
    return values


def apply_config(args: argparse.Namespace) -> None:
    if not args.config:
        return
    supplied = {
        item[2:].replace("-", "_")
        for item in sys.argv[1:]
        if item.startswith("--")
    }
    for key, value in load_simple_yaml(Path(args.config)).items():
        if hasattr(args, key) and key not in supplied:
            setattr(args, key, value)


def pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "by_pair_type": dict(Counter(str(row.get("pair_type", "unknown")) for row in rows)),
        "by_setting": dict(Counter(str(row.get("setting", "unknown")) for row in rows)),
    }


def ensure_run_guards(args: argparse.Namespace) -> None:
    init_adapter = Path(args.init_adapter)
    reference_adapter = Path(args.reference_adapter)
    if args.reference_free:
        raise ValueError("reference_free=true is blocked: DPO-v8 must use a frozen r3 reference")
    if not args.init_adapter or not (init_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"policy init adapter is missing: {init_adapter}")
    if not args.reference_adapter or not (reference_adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"reference adapter is missing: {reference_adapter}")
    if init_adapter.resolve() != reference_adapter.resolve():
        raise ValueError("init_adapter and reference_adapter must both point to the r3 adapter")
    if Path(args.output_dir).resolve() in {init_adapter.resolve(), reference_adapter.resolve()}:
        raise ValueError("DPO output_dir must not overwrite the r3 adapter")


def image_paths(row: dict[str, Any], max_images: int) -> list[str]:
    paths: list[str] = []
    for item in row.get("image_paths") or []:
        path = Path(str(item))
        if path.exists():
            paths.append(path.as_posix())
        if max_images and len(paths) >= max_images:
            break
    return paths


def response_text(row: dict[str, Any], side: str) -> str:
    return json.dumps(row[side], ensure_ascii=False, separators=(",", ":"))


def build_messages(row: dict[str, Any], side: str | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for path in image_paths(row, args.max_images):
        image: dict[str, Any] = {"type": "image", "image": path}
        if args.max_pixels > 0:
            image["max_pixels"] = args.max_pixels
        content.append(image)
    content.append({"type": "text", "text": str(row.get("prompt", ""))})
    messages = [{"role": "user", "content": content}]
    if side:
        messages.append({"role": "assistant", "content": response_text(row, side)})
    return messages


def encode_response(processor: Any, row: dict[str, Any], side: str, device: Any, args: argparse.Namespace) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    prompt_messages = build_messages(row, None, args)
    full_messages = build_messages(row, side, args)
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(full_messages)
    prompt_inputs = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    full_inputs = processor(text=[full_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    labels = full_inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
    labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs["labels"] = labels
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in full_inputs.items()}


def sequence_logprob(model: Any, batch: dict[str, Any]) -> Any:
    import torch.nn.functional as functional

    labels = batch["labels"][:, 1:]
    outputs = model(**{key: value for key, value in batch.items() if key != "labels"}, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    mask = labels.ne(-100)
    selected = labels.masked_fill(~mask, 0)
    token_logps = functional.log_softmax(logits, dim=-1).gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1).mean()


def load_peft_model(args: argparse.Namespace, trainable: bool) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    quantization_config = None
    if args.qlora:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
    kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": dtype, "device_map": "auto" if args.qlora else None}
    if quantization_config:
        kwargs["quantization_config"] = quantization_config
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=args.use_fast_processor)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs)
    if not args.qlora:
        model = model.to("cuda")
    if trainable and args.qlora:
        model = prepare_model_for_kbit_training(model)
    adapter = args.init_adapter if trainable else args.reference_adapter
    model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    model.config.use_cache = False
    if trainable and args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if trainable:
        model.train()
    else:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, processor


def train(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    policy, processor = load_peft_model(args, trainable=True)
    reference, _ = load_peft_model(args, trainable=False)
    device = next(policy.parameters()).device
    optimizer = torch.optim.AdamW((parameter for parameter in policy.parameters() if parameter.requires_grad), lr=args.learning_rate)
    metric_path = Path(args.log_dir) / "dpo_metrics.jsonl"
    loss_path = Path(args.log_dir) / "loss_log.jsonl"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    steps = 0
    last_record: dict[str, Any] = {}
    saved_checkpoints: list[str] = []
    optimizer.zero_grad(set_to_none=True)
    epochs = max(1, math.ceil(args.max_steps * args.gradient_accumulation_steps / max(1, len(rows))))
    with metric_path.open("w", encoding="utf-8") as metrics, loss_path.open("w", encoding="utf-8") as losses:
        accum: list[dict[str, float]] = []
        for epoch in range(epochs):
            for index, row in enumerate(rows):
                chosen = encode_response(processor, row, "chosen", device, args)
                rejected = encode_response(processor, row, "rejected", device, args)
                policy_chosen = sequence_logprob(policy, chosen)
                policy_rejected = sequence_logprob(policy, rejected)
                with torch.inference_mode():
                    ref_chosen = sequence_logprob(reference, chosen)
                    ref_rejected = sequence_logprob(reference, rejected)
                chosen_reward = args.beta * (policy_chosen - ref_chosen)
                rejected_reward = args.beta * (policy_rejected - ref_rejected)
                reward_margin = chosen_reward - rejected_reward
                loss = -functional.logsigmoid(reward_margin)
                raw_loss = float(loss.detach().cpu())
                raw_margin = float(reward_margin.detach().cpu())
                if not math.isfinite(raw_loss) or not math.isfinite(raw_margin):
                    raise FloatingPointError(f"NaN/Inf DPO metric at epoch={epoch + 1} row={index + 1}")
                (loss / args.gradient_accumulation_steps).backward()
                accum.append({
                    "loss": raw_loss,
                    "chosen_logprob": float(policy_chosen.detach().cpu()),
                    "rejected_logprob": float(policy_rejected.detach().cpu()),
                    "policy_logratio": float((policy_chosen - policy_rejected).detach().cpu()),
                    "reference_logratio": float((ref_chosen - ref_rejected).detach().cpu()),
                    "logratio_delta": float(((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)).detach().cpu()),
                    "chosen_reward": float(chosen_reward.detach().cpu()),
                    "rejected_reward": float(rejected_reward.detach().cpu()),
                    "reward_margin": raw_margin,
                })
                update = (index + 1) % args.gradient_accumulation_steps == 0 or index + 1 == len(rows)
                if not update:
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                last_record = {
                    "epoch": epoch + 1,
                    "step": steps,
                    "loss": sum(item["loss"] for item in accum) / len(accum),
                    "chosen_logprob": sum(item["chosen_logprob"] for item in accum) / len(accum),
                    "rejected_logprob": sum(item["rejected_logprob"] for item in accum) / len(accum),
                    "policy_logratio": sum(item["policy_logratio"] for item in accum) / len(accum),
                    "reference_logratio": sum(item["reference_logratio"] for item in accum) / len(accum),
                    "logratio_delta": sum(item["logratio_delta"] for item in accum) / len(accum),
                    "chosen_reward": sum(item["chosen_reward"] for item in accum) / len(accum),
                    "rejected_reward": sum(item["rejected_reward"] for item in accum) / len(accum),
                    "reward_margin": sum(item["reward_margin"] for item in accum) / len(accum),
                    "beta": args.beta,
                    "learning_rate": args.learning_rate,
                }
                accum = []
                metrics.write(json.dumps(last_record, ensure_ascii=False) + "\n")
                metrics.flush()
                losses.write(json.dumps({"epoch": epoch + 1, "step": steps, "loss": last_record["loss"]}, ensure_ascii=False) + "\n")
                losses.flush()
                print(json.dumps(last_record, ensure_ascii=False), flush=True)
                if args.save_steps and steps % args.save_steps == 0:
                    checkpoint = Path(args.output_dir) / f"checkpoint-step-{steps:06d}"
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(checkpoint)
                    processor.save_pretrained(checkpoint)
                    saved_checkpoints.append(checkpoint.as_posix())
                if steps >= args.max_steps:
                    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                    policy.save_pretrained(args.output_dir)
                    processor.save_pretrained(args.output_dir)
                    return {
                        "success": True,
                        "steps": steps,
                        "final_loss": last_record["loss"],
                        "final_reward_margin": last_record["reward_margin"],
                        "saved_checkpoints": saved_checkpoints,
                    }
    raise RuntimeError(f"training stopped at {steps} steps before max_steps={args.max_steps}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO-v8 LoRA smoke training from frozen r3 reference.")
    parser.add_argument("--config", default="")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--init_adapter", default="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/")
    parser.add_argument("--reference_adapter", default="checkpoints/qwen25vl_lora_sft_v3_r3_lingo_smoke/")
    parser.add_argument("--preference_file", default="data/train/preference_v8/preference_v8_pairs.jsonl")
    parser.add_argument("--output_dir", default="checkpoints/qwen25vl_lora_dpo_v8_lingo_smoke/")
    parser.add_argument("--log_dir", default="outputs/train_logs/dpo_v8_lingo_smoke/")
    parser.add_argument("--reference_free", action="store_true")
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=2e-7)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_fast_processor", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    apply_config(args)
    return args


def main() -> None:
    args = parse_args()
    ensure_run_guards(args)
    if not args.dry_run and (Path(args.output_dir) / "adapter_config.json").exists():
        raise FileExistsError(f"refusing to overwrite an existing DPO-v8 adapter: {args.output_dir}")
    rows = read_jsonl(Path(args.preference_file), args.train_samples)
    random.Random(args.seed).shuffle(rows)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir / "train_config.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in sorted(vars(args).items())) + "\n",
        encoding="utf-8",
    )
    write_json(log_dir / "dataset_stats.json", pair_stats(rows))
    (log_dir / "command.txt").write_text(" ".join(shlex.quote(arg) for arg in sys.argv) + "\n", encoding="utf-8")
    summary: dict[str, Any] = {
        "success": False,
        "dry_run": bool(args.dry_run),
        "init_adapter": args.init_adapter,
        "reference_adapter": args.reference_adapter,
        "reference_free": bool(args.reference_free),
        "output_dir": args.output_dir,
        "max_steps": args.max_steps,
        "save_steps": args.save_steps,
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "final_loss": None,
        "final_reward_margin": None,
        "nan_detected": False,
        "oom": False,
        "saved_checkpoints": [],
    }
    if args.dry_run:
        (log_dir / "loss_log.jsonl").write_text("", encoding="utf-8")
        (log_dir / "dpo_metrics.jsonl").write_text("", encoding="utf-8")
        summary.update({"success": True, "rows": len(rows)})
        write_json(log_dir / "train_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    try:
        summary.update(train(args, rows))
    except FloatingPointError as exc:
        summary["nan_detected"] = True
        summary["error"] = str(exc)
        write_json(log_dir / "train_summary.json", summary)
        raise
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            summary["oom"] = True
            summary["suggestion"] = "lower max_pixels, max sequence length, or batch size"
        summary["error"] = str(exc)
        write_json(log_dir / "train_summary.json", summary)
        raise
    except Exception as exc:
        summary["error"] = str(exc)
        write_json(log_dir / "train_summary.json", summary)
        raise
    write_json(log_dir / "train_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
