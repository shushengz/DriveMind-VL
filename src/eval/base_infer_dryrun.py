"""Dry-run inference that emits deterministic dummy predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                if line.strip():
                    rows.append(json.loads(line))
            except Exception as exc:
                print(f"skip line {line_no}: {exc}")
    return rows


def perturb_prediction(sample: dict[str, Any], idx: int) -> dict[str, Any] | str:
    gold = dict(sample.get("answer", {}))
    task = sample.get("meta", {}).get("task_type")
    if idx % 10 == 0:
        return "not a json output"
    if idx % 4 != 0:
        return gold
    pred = dict(gold)
    if task == "risk_reasoning":
        pred["risk_level"] = "low" if gold.get("risk_level") != "low" else "high"
        pred["suggestion"] = "keep_speed"
    elif task in {"tool_call", "personalized_service"}:
        pred["tool"] = "play_music" if gold.get("tool") != "play_music" else "set_ac_temperature"
        pred["arguments"] = {}
    elif task == "safety_rejection":
        pred = {"task": "tool_call", "tool": gold.get("tool", "open_door"), "arguments": gold.get("arguments", {}), "reason": "执行用户请求"}
    elif task == "cabin_understanding":
        pred["driver_state"] = "normal" if gold.get("driver_state") != "normal" else "fatigued"
    return pred


def run(input_path: Path, output_path: Path) -> int:
    samples = load_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            row = {
                "id": sample.get("id"),
                "prediction": perturb_prediction(sample, idx),
                "gold": sample.get("answer", {}),
                "vehicle_state": sample.get("vehicle_state", {}),
                "meta": sample.get("meta", {}),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DriveMind-VL dry-run inference.")
    parser.add_argument("--input", default="data/processed/drivemind_seed.jsonl")
    parser.add_argument("--output", default="outputs/eval_results/base_predictions.jsonl")
    parser.add_argument("--dry_run", action="store_true", default=True)
    parser.add_argument("--model_name_or_path", default="", help="Reserved for future real-model inference. Not used by default.")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("Real model inference is not enabled in the local MVP. Use --dry_run.")
    count = run(Path(args.input), Path(args.output))
    print(f"wrote {count} dry-run predictions to {args.output}")


if __name__ == "__main__":
    main()

