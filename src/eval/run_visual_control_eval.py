"""Run strict visual-control eval for Qwen2.5-VL or LoRA adapters.

Default dry-run writes deterministic predictions and never loads a model.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics_visual_control import answer_length, exact_match, is_control_hallucination, is_refusal, text_content, token_f1

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")


def selected_settings(value: str) -> tuple[str, ...]:
    if not value:
        return SETTINGS
    chosen = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [item for item in chosen if item not in SETTINGS]
    if invalid or not chosen:
        raise ValueError(f"--settings must be a comma-separated subset of {SETTINGS}; received {value!r}")
    return chosen


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_requested_ids(path: str) -> list[str] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list):
        raise ValueError(f"--eval_ids must contain a JSON list or an object with 'ids': {path}")
    requested = [str(sample_id) for sample_id in ids if str(sample_id)]
    if not requested:
        raise ValueError(f"--eval_ids selects zero cases: {path}. Build a non-empty held-out set before evaluation.")
    if len(requested) != len(set(requested)):
        raise ValueError(f"--eval_ids contains duplicate IDs: {path}")
    return requested


def read_aligned_setting_samples(args: argparse.Namespace, dataset: str) -> dict[str, list[dict[str, Any]]]:
    rows_by_setting: dict[str, list[dict[str, Any]]] = {}
    maps_by_setting: dict[str, dict[str, dict[str, Any]]] = {}
    for setting in SETTINGS:
        input_path = Path(args.visual_control_dir) / f"{dataset}_strict_{setting}.jsonl"
        rows = read_jsonl(input_path)
        rows_by_setting[setting] = rows
        maps_by_setting[setting] = {str(row.get("id")): row for row in rows if row.get("id")}

    common_ids = set(maps_by_setting["normal"])
    for setting in SETTINGS[1:]:
        common_ids &= set(maps_by_setting[setting])

    if not common_ids:
        counts = {setting: len(rows_by_setting[setting]) for setting in SETTINGS}
        raise ValueError(f"No aligned visual-control ids for {dataset}; setting counts={counts}")

    requested_ids = load_requested_ids(args.eval_ids)
    if requested_ids is not None:
        missing = [sample_id for sample_id in requested_ids if sample_id not in common_ids]
        if missing:
            raise ValueError(f"--eval_ids includes IDs missing from one or more settings: {missing[:5]}")
        ordered_ids = requested_ids
    else:
        ordered_ids = [str(row.get("id")) for row in rows_by_setting["normal"] if str(row.get("id")) in common_ids]
    if args.eval_samples and requested_ids is None:
        ordered_ids = ordered_ids[: args.eval_samples]
    if args.eval_samples and requested_ids is None and len(ordered_ids) < args.eval_samples:
        print(
            f"[warn] requested {args.eval_samples} aligned cases for {dataset}, "
            f"but only {len(ordered_ids)} are available.",
            file=sys.stderr,
        )
    return {setting: [maps_by_setting[setting][sample_id] for sample_id in ordered_ids] for setting in SETTINGS}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_answer(prediction: str) -> str:
    return text_content(prediction).strip()


def make_prompt(sample: dict[str, Any], setting: str) -> str:
    control_note = {
        "normal": "Answer the driving visual question using only the supplied visual input.",
        "text_only": "No image is supplied. If the question needs visual evidence, say the image information is insufficient.",
        "wrong_image": "The supplied image may not match the question. Do not infer the original scene unless the visual evidence supports it.",
        "blank_image": "The supplied image may be blank. If there is no usable visual evidence, say the image information is insufficient.",
    }[setting]
    return (
        "You are DriveMind-VL. Return ONLY one JSON object with keys answer and reason. "
        "Keep answer concise. Do not mention dataset paths. "
        f"{control_note}\n"
        f"[Question]\n{sample.get('question','')}\n\n"
        f"[Visual labels]\n{' '.join(sample.get('image_labels') or [])}"
    )


def dry_prediction(sample: dict[str, Any], setting: str, index: int) -> str:
    gold = str(sample.get("gold") or "")
    if setting == "normal":
        answer = gold if index % 3 else "unknown"
        reason = "dry-run normal visual placeholder"
    elif setting == "text_only":
        answer = "缺少图像信息，无法可靠判断。"
        reason = "dry-run text-only calibration placeholder"
    elif setting == "blank_image":
        answer = "当前图像信息不足，无法确定。"
        reason = "dry-run blank-image calibration placeholder"
    else:
        answer = "不应基于该图像确定原问题答案。"
        reason = "dry-run wrong-image calibration placeholder"
    return json.dumps({"answer": answer, "reason": reason}, ensure_ascii=False)


def load_model(args: argparse.Namespace):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quantization_config = None
    if args.load_in_4bit or args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": args.device_map, "dtype": torch.bfloat16 if args.bf16 else "auto"}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    processor_path = args.adapter_path if args.adapter_path else args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True, use_fast=args.use_fast_processor)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    return model, processor


def generate_real(model: Any, processor: Any, sample: dict[str, Any], setting: str, args: argparse.Namespace) -> str:
    import torch
    from qwen_vl_utils import process_vision_info

    content: list[dict[str, Any]] = []
    for path in sample.get("image_paths") or []:
        p = Path(str(path))
        if p.exists():
            item: dict[str, Any] = {"type": "image", "image": p.as_posix()}
            if args.max_pixels > 0:
                item["max_pixels"] = args.max_pixels
            content.append(item)
    content.append({"type": "text", "text": make_prompt(sample, setting)})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    device = getattr(model, "device", None) or next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(generated[:, input_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def score_row(sample: dict[str, Any], prediction: str, model_name: str, setting: str) -> dict[str, Any]:
    pred_answer = parse_answer(prediction)
    gold = str(sample.get("gold") or "")
    return {
        "id": sample.get("id", ""),
        "dataset": sample.get("dataset", ""),
        "model_name": model_name,
        "mode": sample.get("mode", "strict_visual"),
        "setting": setting,
        "question": sample.get("question", ""),
        "image_paths": sample.get("image_paths", []),
        "image_labels": sample.get("image_labels", []),
        "gold": gold,
        "prediction": prediction,
        "f1": token_f1(pred_answer, gold),
        "em": exact_match(pred_answer, gold),
        "answer_length": answer_length(pred_answer),
        "is_refusal": is_refusal(prediction),
        "is_visual_hallucination": is_control_hallucination(prediction, setting),
    }


def run_dataset(args: argparse.Namespace, dataset: str) -> dict[str, Path]:
    aligned_samples = read_aligned_setting_samples(args, dataset)
    target_settings = selected_settings(args.settings)
    model = processor = None
    if not args.dry_run:
        model, processor = load_model(args)
    prediction_root = Path(args.prediction_root) if args.prediction_root else Path(args.output_root) / "predictions"
    out_dir = prediction_root / dataset / args.model_name / "strict_visual"
    all_rows: list[dict[str, Any]] = []
    outputs: dict[str, Path] = {}
    for setting in target_settings:
        rows = aligned_samples[setting]
        pred_rows = []
        for idx, sample in enumerate(rows):
            prediction = dry_prediction(sample, setting, idx) if args.dry_run else generate_real(model, processor, sample, setting, args)
            pred_rows.append(score_row(sample, prediction, args.model_name, setting))
        output_path = out_dir / f"{setting}.jsonl"
        write_jsonl(output_path, pred_rows)
        outputs[setting] = output_path
        all_rows.extend(pred_rows)
    if target_settings == SETTINGS:
        combined = out_dir / "all_settings.jsonl"
        write_jsonl(combined, all_rows)
        outputs["all_settings"] = combined
        cmd = [sys.executable, "src/eval/summarize_visual_control.py", "--input", combined.as_posix(), "--output_dir", args.output_root, "--dataset", dataset, "--model_name", args.model_name, "--mode", "strict_visual"]
        if args.dry_run:
            cmd.append("--dry_run")
        subprocess.run(cmd, check=True)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run strict visual-control eval for Stage 2.")
    parser.add_argument("--dataset", choices=["lingoqa", "drivelm", "both"], default="lingoqa")
    parser.add_argument("--visual_control_dir", default="data/processed/visual_control")
    parser.add_argument("--model_path", default="/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--model_name", default="base_qwen25vl_3b")
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--prediction_root", default="")
    parser.add_argument("--eval_ids", default="")
    parser.add_argument("--mode", choices=["strict_visual"], default="strict_visual")
    parser.add_argument("--settings", default="", help="Optional comma-separated settings subset for resumable prediction generation.")
    parser.add_argument("--eval_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--max_pixels", type=int, default=200704)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--use_fast_processor", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = ["lingoqa", "drivelm"] if args.dataset == "both" else [args.dataset]
    summary = {dataset: {key: path.as_posix() for key, path in run_dataset(args, dataset).items()} for dataset in datasets}
    print(json.dumps({"dry_run": args.dry_run, "outputs": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
