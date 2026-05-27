"""Minimal Qwen2.5-VL LoRA DPO trainer for LingoQA preference pairs.

This script intentionally avoids TRL so it can run in the existing
DriveMind-VL server environment with only torch, transformers, peft, and
qwen_vl_utils installed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.base_infer_qwen25vl import make_prompt, sample_input_mode, select_frame_paths


def progress_iter(iterable: Any, total: int, desc: str) -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def read_jsonl(path: Path, max_pairs: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if max_pairs is not None and len(rows) >= max_pairs:
                break
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid jsonl") from exc
    return rows


def answer_json(answer: dict[str, Any]) -> str:
    return json.dumps(answer, ensure_ascii=False, separators=(",", ":"))


def sample_image_paths(sample: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    mode = sample.get("meta", {}).get("train_input_mode") or sample.get("train_input_mode")
    if args.text_only or mode == "text_only":
        return []

    paths: list[Path] = []
    external = sample.get("meta", {}).get("external", {})
    if args.use_all_images and isinstance(external, dict):
        for item in external.get("image_paths", []):
            path = Path(str(item))
            if path.exists():
                paths.append(path)
        paths = select_frame_paths(paths, args.frame_strategy, args.max_images)

    if not paths:
        image_path = Path(str(sample.get("image", "")))
        if image_path.exists():
            paths.append(image_path)
    return paths


def build_messages(sample: dict[str, Any], assistant: dict[str, Any] | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in sample_image_paths(sample, args):
        item: dict[str, Any] = {"type": "image", "image": image_path.as_posix()}
        if args.max_pixels > 0:
            item["max_pixels"] = args.max_pixels
        content.append(item)
    content.append(
        {
            "type": "text",
            "text": make_prompt(
                sample,
                args.prompt_variant,
                perception_mode=args.perception_mode,
                input_mode=sample_input_mode(sample, args.text_only),
            ),
        }
    )
    messages = [{"role": "user", "content": content}]
    if assistant is not None:
        messages.append({"role": "assistant", "content": answer_json(assistant)})
    return messages


def encode_completion(
    processor: Any,
    sample: dict[str, Any],
    assistant: dict[str, Any],
    device: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    prompt_messages = build_messages(sample, assistant=None, args=args)
    full_messages = build_messages(sample, assistant=assistant, args=args)

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


def completion_logprob(model: Any, batch: dict[str, Any], length_normalize: bool) -> Any:
    import torch

    labels = batch.pop("labels")
    outputs = model(**batch)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_log_probs = (token_log_probs * mask).sum(dim=-1)
    if length_normalize:
        seq_log_probs = seq_log_probs / mask.sum(dim=-1).clamp_min(1)
    return seq_log_probs


def safe_log_odds(logp: Any) -> Any:
    import torch

    prob = torch.exp(logp).clamp(max=1 - 1e-6)
    return logp - torch.log1p(-prob)


def sft_anchor_weight(pair: dict[str, Any], args: argparse.Namespace) -> float:
    if "sft_anchor_weight" in pair:
        return float(pair["sft_anchor_weight"])
    meta = pair.get("meta", {}) if isinstance(pair.get("meta"), dict) else {}
    if "sft_anchor_weight" in meta:
        return float(meta["sft_anchor_weight"])
    mode = str(pair.get("train_input_mode", "") or meta.get("train_input_mode", ""))
    pair_type = str(pair.get("pair_type", "") or meta.get("pair_type", ""))
    anchor_pair_types = {item.strip() for item in args.sft_anchor_pair_types.split(",") if item.strip()}
    if mode == "normal" or pair_type in anchor_pair_types:
        return float(args.default_normal_sft_anchor_weight)
    return float(args.default_control_sft_anchor_weight)


def dataset_summary(pairs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    by_mode = Counter(str(pair.get("train_input_mode", "")) for pair in pairs)
    by_type = Counter(str(pair.get("pair_type", "")) for pair in pairs)
    sft_weight_by_mode: dict[str, float] = defaultdict(float)
    pair_weight_by_mode: dict[str, float] = defaultdict(float)
    for pair in pairs:
        mode = str(pair.get("train_input_mode", ""))
        sft_weight_by_mode[mode] += sft_anchor_weight(pair, args)
        pair_weight_by_mode[mode] += float(pair.get("weight", pair.get("meta", {}).get("weight", 1.0)))
    return {
        "pairs": len(pairs),
        "by_train_input_mode": dict(by_mode),
        "by_pair_type": dict(by_type),
        "sft_anchor_weight_sum_by_mode": {key: round(value, 4) for key, value in sorted(sft_weight_by_mode.items())},
        "pair_weight_sum_by_mode": {key: round(value, 4) for key, value in sorted(pair_weight_by_mode.items())},
    }


def save_adapter(model: Any, processor: Any, output_dir: Path, name: str) -> Path:
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def pair_loss(model: Any, pair: dict[str, Any], processor: Any, device: str, args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as F

    sample = pair["prompt_sample"]
    sample.setdefault("meta", {})
    sample["meta"]["train_input_mode"] = pair.get("train_input_mode", "normal")

    chosen_batch = encode_completion(processor, sample, pair["chosen"], device, args)
    rejected_batch = encode_completion(processor, sample, pair["rejected"], device, args)

    pi_chosen = completion_logprob(model, dict(chosen_batch), args.length_normalize)
    pi_rejected = completion_logprob(model, dict(rejected_batch), args.length_normalize)

    if args.reference_free or args.loss_type in {"simpo", "orpo"}:
        ref_chosen = torch.zeros_like(pi_chosen)
        ref_rejected = torch.zeros_like(pi_rejected)
    else:
        with torch.no_grad():
            with model.disable_adapter():
                ref_chosen = completion_logprob(model, dict(chosen_batch), args.length_normalize)
                ref_rejected = completion_logprob(model, dict(rejected_batch), args.length_normalize)

    pi_margin = pi_chosen - pi_rejected
    ref_margin = ref_chosen - ref_rejected
    if args.loss_type == "simpo":
        dpo_loss = -F.logsigmoid(args.beta * (pi_margin - args.simpo_gamma)).mean()
    elif args.loss_type == "orpo":
        odds_margin = safe_log_odds(pi_chosen) - safe_log_odds(pi_rejected)
        dpo_loss = -F.logsigmoid(odds_margin).mean()
    else:
        dpo_loss = -F.logsigmoid(args.beta * (pi_margin - ref_margin)).mean()
    chosen_sft_loss = -pi_chosen.mean()
    pair_weight = float(pair.get("weight", pair.get("meta", {}).get("weight", 1.0)))
    anchor_weight = sft_anchor_weight(pair, args)
    loss = pair_weight * dpo_loss + args.chosen_sft_weight * anchor_weight * chosen_sft_loss

    stats = {
        "loss": float(loss.detach().cpu()),
        "dpo_loss": float(dpo_loss.detach().cpu()),
        "chosen_sft_loss": float(chosen_sft_loss.detach().cpu()),
        "pi_margin": float(pi_margin.detach().mean().cpu()),
        "ref_margin": float(ref_margin.detach().mean().cpu()),
        "pair_weight": pair_weight,
        "sft_anchor_weight": anchor_weight,
    }
    return loss, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal Qwen2.5-VL LoRA DPO training.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--init_adapter_path",
        default="",
        help="Optional existing LoRA adapter to continue from, for example an SFT-v2 adapter.",
    )
    parser.add_argument("--train_file", default="data/processed/lingoqa_sft_v1_preference_pairs_77.jsonl")
    parser.add_argument("--output_dir", default="outputs/checkpoints/qwen25vl_3b_lingoqa_sft_v1_dpo_pref77_smoke")
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target modules used when --init_adapter_path is empty.",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--chosen_sft_weight", type=float, default=0.05)
    parser.add_argument(
        "--sft_anchor_pair_types",
        default="normal_answer_over_refusal,normal_answer_over_bad_prediction",
        help="Comma-separated pair types that receive the default normal SFT anchor weight.",
    )
    parser.add_argument("--default_normal_sft_anchor_weight", type=float, default=1.0)
    parser.add_argument("--default_control_sft_anchor_weight", type=float, default=0.0)
    parser.add_argument("--loss_type", choices=["dpo", "simpo", "orpo"], default="dpo")
    parser.add_argument("--simpo_gamma", type=float, default=0.0)
    parser.add_argument("--length_normalize", action="store_true", default=True)
    parser.add_argument("--no_length_normalize", dest="length_normalize", action="store_false")
    parser.add_argument("--reference_free", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=20260517)
    parser.add_argument("--max_steps", type=int, default=0, help="Stop after this many optimizer steps. 0 means full epochs.")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint-step-XXXXXX every N optimizer steps.")
    parser.add_argument("--use_all_images", action="store_true")
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument(
        "--frame_strategy",
        default="first_middle_last",
        choices=["first_n", "first", "middle", "last", "first_middle_last", "uniform"],
    )
    parser.add_argument(
        "--prompt_variant",
        default="spatial",
        choices=["current", "temporal", "spatial", "evidence"],
    )
    parser.add_argument("--perception_mode", default="full", choices=["full", "none"])
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--text_only", action="store_true")
    parser.add_argument("--use_fast_processor", action="store_true")
    args = parser.parse_args()

    max_pairs = args.max_pairs if args.max_pairs and args.max_pairs > 0 else None
    pairs = read_jsonl(Path(args.train_file), max_pairs=max_pairs)
    if args.shuffle_seed >= 0:
        random.Random(args.shuffle_seed).shuffle(pairs)
        print(f"shuffled preference pairs with seed={args.shuffle_seed}", flush=True)
    print(f"loaded preference pairs: {len(pairs)}", flush=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dataset_summary(pairs, args)
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        print("dry_run: skip model loading and training", flush=True)
        return

    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16 if torch.cuda.is_available() else torch.float32
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=args.use_fast_processor,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=None,
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False

    if args.init_adapter_path:
        print(f"loading trainable init adapter: {args.init_adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
    else:
        target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    full_total_steps = args.epochs * math.ceil(len(pairs) / args.gradient_accumulation_steps)
    total_steps = min(full_total_steps, args.max_steps) if args.max_steps and args.max_steps > 0 else full_total_steps
    print(
        f"epochs={args.epochs} optimizer_steps~={total_steps} "
        f"loss_type={args.loss_type} beta={args.beta} simpo_gamma={args.simpo_gamma} "
        f"chosen_sft_weight={args.chosen_sft_weight}",
        flush=True,
    )

    global_step = 0
    stop_training = False
    log_path = output_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        epoch_bar = progress_iter(pairs, total=len(pairs), desc=f"dpo epoch {epoch + 1}/{args.epochs}")
        for idx, pair in enumerate(epoch_bar):
            loss, stats = pair_loss(model, pair, processor, device, args)
            (loss / args.gradient_accumulation_steps).backward()
            if (idx + 1) % args.gradient_accumulation_steps == 0 or idx + 1 == len(pairs):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if hasattr(epoch_bar, "set_postfix"):
                    epoch_bar.set_postfix(
                        step=global_step,
                        loss=f"{stats['loss']:.4f}",
                        dpo=f"{stats['dpo_loss']:.4f}",
                        w=f"{stats['pair_weight']:.2f}",
                        sft_w=f"{stats['sft_anchor_weight']:.2f}",
                    )
                mode = str(pair.get("train_input_mode", ""))
                pair_type = str(pair.get("pair_type", ""))
                log_row = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "pair_id": pair.get("id"),
                    "train_input_mode": mode,
                    "pair_type": pair_type,
                    **stats,
                }
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={stats['loss']:.4f} dpo={stats['dpo_loss']:.4f} "
                    f"chosen_sft={stats['chosen_sft_loss']:.4f} "
                    f"pi_margin={stats['pi_margin']:.4f} ref_margin={stats['ref_margin']:.4f} "
                    f"mode={mode} type={pair_type} "
                    f"pair_w={stats['pair_weight']:.2f} sft_w={stats['sft_anchor_weight']:.2f}",
                    flush=True,
                )
                if args.save_steps and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_dir = save_adapter(model, processor, output_dir, f"checkpoint-step-{global_step:06d}")
                    print(f"saved checkpoint to {checkpoint_dir}", flush=True)
                if args.max_steps and args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break
        if stop_training:
            break

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"saved LoRA adapter to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
