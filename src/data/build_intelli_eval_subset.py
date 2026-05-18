"""Build a balanced DriveMind external eval subset from IntelliCockpitBench JSONL."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.audit_intelli_cockpitbench import resolve_image_path
from src.data.convert_external_to_drivemind import convert_record, dump_jsonl, load_records
from src.data.external_vqa_taxonomy import infer_external_vqa_capability, normalize_reference


def capability_of_raw(record: dict[str, Any]) -> str:
    return infer_external_vqa_capability(
        instruction=str(record.get("question") or record.get("instruction") or ""),
        category=str(record.get("category") or ""),
        subcategory=str(record.get("subcategory") or record.get("sub_category") or ""),
        reference=normalize_reference(record.get("reference") or record.get("answer")),
    )


def filter_records(records: list[dict[str, Any]], image_root: Path | None, require_images: bool) -> list[dict[str, Any]]:
    if not require_images:
        return records
    filtered = []
    for record in records:
        image_path = resolve_image_path(record, image_root)
        if image_path is not None and image_path.exists():
            filtered.append(record)
    return filtered


def balanced_sample(records: list[dict[str, Any]], target_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[capability_of_raw(record)].append(record)
    for group in groups.values():
        rng.shuffle(group)

    capabilities = sorted(groups)
    selected: list[dict[str, Any]] = []
    cursor = {capability: 0 for capability in capabilities}
    while len(selected) < target_size and capabilities:
        progressed = False
        for capability in capabilities:
            idx = cursor[capability]
            if idx < len(groups[capability]) and len(selected) < target_size:
                selected.append(groups[capability][idx])
                cursor[capability] += 1
                progressed = True
        if not progressed:
            break
    return selected


def summarize_raw(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"total": len(records), "capability_counts": {}}
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[capability_of_raw(record)] += 1
    summary["capability_counts"] = dict(sorted(counts.items()))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced DriveMind subset from IntelliCockpitBench JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output", default="data/processed/drivemind_intelli_eval_subset.jsonl")
    parser.add_argument("--target_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="eval", choices=["train", "val", "eval", "test"])
    parser.add_argument("--allow_missing_images", action="store_true")
    parser.add_argument("--manifest_output", default="outputs/eval_results/intelli_eval_subset_manifest.json")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    image_root = Path(args.image_root)
    usable = filter_records(records, image_root, require_images=not args.allow_missing_images)
    selected = balanced_sample(usable, target_size=args.target_size, seed=args.seed)
    converted = [
        convert_record(record, index=index, source="intelli_cockpit_bench", image_root=image_root, split=args.split)
        for index, record in enumerate(selected, start=1)
    ]
    dump_jsonl(converted, Path(args.output))

    manifest = {
        "input": args.input,
        "image_root": args.image_root,
        "target_size": args.target_size,
        "seed": args.seed,
        "source_total": len(records),
        "usable_with_images": len(usable),
        "selected": summarize_raw(selected),
    }
    Path(args.manifest_output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.manifest_output).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"wrote subset to {args.output}")
    print(f"wrote manifest to {args.manifest_output}")


if __name__ == "__main__":
    main()
