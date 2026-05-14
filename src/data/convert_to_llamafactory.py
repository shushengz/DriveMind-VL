"""Convert DriveMind-Instruct JSONL to a common multimodal SFT format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"skip line {line_no}: {exc}")
    return rows


def make_user_content(sample: dict[str, Any]) -> str:
    vehicle_state = json.dumps(sample.get("vehicle_state", {}), ensure_ascii=False)
    perception = json.dumps(sample.get("perception", {}), ensure_ascii=False)
    return (
        "你是车载智能座舱助手。请根据图像、车辆状态和结构化感知信息完成任务。\n\n"
        f"[Instruction]\n{sample.get('instruction', '')}\n\n"
        f"[Vehicle State]\n{vehicle_state}\n\n"
        f"[Perception]\n{perception}"
    )


def convert_sample(sample: dict[str, Any]) -> dict[str, Any]:
    answer = sample.get("answer", {})
    return {
        "messages": [
            {"role": "user", "content": make_user_content(sample)},
            {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False, separators=(",", ":"))},
        ],
        "images": [sample.get("image", "")],
    }


def convert(input_path: Path, output_path: Path) -> list[dict[str, Any]]:
    rows = [convert_sample(sample) for sample in load_jsonl(input_path)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert DriveMind data to multimodal SFT JSON.")
    parser.add_argument("--input", default="data/processed/drivemind_seed.jsonl")
    parser.add_argument("--output", default="data/processed/drivemind_llamafactory_sft.json")
    args = parser.parse_args()
    rows = convert(Path(args.input), Path(args.output))
    print(f"wrote {len(rows)} records to {args.output}")


if __name__ == "__main__":
    main()

