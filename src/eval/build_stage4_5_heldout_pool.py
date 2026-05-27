"""Construct a strict LingoQA held-out pool from an untouched clean split."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.visual_control_formatter import MODE_STRICT, SETTINGS, build_dataset, output_stem, read_jsonl, write_jsonl


def source_question_id(row: dict[str, Any]) -> str:
    external = ((row.get("meta") or {}).get("external") or {})
    return str(external.get("question_id") or (row.get("metadata") or {}).get("original_id") or row.get("id") or "")


def existing_original_ids(path: Path) -> set[str]:
    return {str((row.get("metadata") or {}).get("original_id")) for row in read_jsonl(path) if (row.get("metadata") or {}).get("original_id")}


def build_pool(source_file: Path, train_normal_file: Path, output_dir: Path, seed: int, blank_image: str, max_samples: int = 0) -> dict[str, Any]:
    source_rows = read_jsonl(source_file, max_samples)
    train_original = existing_original_ids(train_normal_file)
    kept: list[dict[str, Any]] = []
    rejected: list[str] = []
    for row in source_rows:
        original_id = source_question_id(row)
        if original_id in train_original:
            rejected.append(original_id)
            continue
        updated = deepcopy(row)
        updated["id"] = f"lingoqa_test_{original_id}"
        kept.append(updated)
    outputs = build_dataset(kept, "lingoqa", MODE_STRICT, seed, blank_image=blank_image or None)
    written: dict[str, Any] = {}
    for setting in SETTINGS:
        path = output_dir / output_stem("lingoqa", MODE_STRICT, setting)
        written[setting] = {"path": path.as_posix(), "count": write_jsonl(path, outputs[setting])}
    result = {
        "source_file": source_file.as_posix(),
        "train_normal_file": train_normal_file.as_posix(),
        "source_rows": len(source_rows),
        "train_original_ids": len(train_original),
        "rejected_overlap_original_ids": rejected,
        "rejected_overlap_count": len(rejected),
        "heldout_pool_count": len(kept),
        "source_original_id_disjoint": not rejected,
        "outputs": written,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "heldout_pool_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict held-out pool from the LingoQA clean test split.")
    parser.add_argument("--source_file", default="data/processed/lingoqa_clean_v2_test.jsonl")
    parser.add_argument("--train_normal_file", default="data/processed/visual_control/lingoqa_strict_normal.jsonl")
    parser.add_argument("--output_dir", default="data/processed/visual_control_heldout")
    parser.add_argument("--blank_image", default="outputs/cases/ablation_images/blank.jpg")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_pool(Path(args.source_file), Path(args.train_normal_file), Path(args.output_dir), args.seed, args.blank_image, args.max_samples)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["rejected_overlap_count"]:
        raise SystemExit("held-out source contains training original IDs; pool contains only disjoint retained rows")


if __name__ == "__main__":
    main()
