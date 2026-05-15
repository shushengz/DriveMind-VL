"""Prepare a tiny IntelliCockpitBench sample in DriveMind-Instruct format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.convert_external_to_drivemind import convert_record, dump_jsonl, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare IntelliCockpitBench sample JSONL for DriveMind external eval.")
    parser.add_argument("--repo_root", default="third_party/IntelliCockpitBench")
    parser.add_argument("--input", default="", help="Override input JSONL path.")
    parser.add_argument("--image_root", default="", help="Override image root path.")
    parser.add_argument("--output", default="data/processed/drivemind_intelli_sample.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", default="eval", choices=["train", "val", "eval", "test"])
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    input_path = Path(args.input) if args.input else repo_root / "Evaluation" / "data" / "jsonl" / "english_test.jsonl"
    image_root = Path(args.image_root) if args.image_root else repo_root / "Evaluation" / "data" / "images"
    if not input_path.exists():
        raise SystemExit(f"IntelliCockpitBench sample JSONL not found: {input_path}")
    if not image_root.exists():
        raise SystemExit(f"IntelliCockpitBench image directory not found: {image_root}")

    records = load_records(input_path)
    if args.limit > 0:
        records = records[: args.limit]
    converted = [
        convert_record(
            record,
            index=index,
            source="intelli_cockpit_bench",
            image_root=image_root,
            split=args.split,
        )
        for index, record in enumerate(records, start=1)
    ]
    dump_jsonl(converted, Path(args.output))
    print(f"prepared {len(converted)} IntelliCockpitBench samples to {args.output}")


if __name__ == "__main__":
    main()
