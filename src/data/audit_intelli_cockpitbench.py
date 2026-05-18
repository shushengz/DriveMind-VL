"""Audit an IntelliCockpitBench-style JSONL before building an eval subset."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.convert_external_to_drivemind import load_records
from src.data.external_vqa_taxonomy import infer_external_vqa_capability, normalize_reference


def resolve_image_path(record: dict[str, Any], image_root: Path | None) -> Path | None:
    image_value = record.get("img_path") or record.get("image") or record.get("image_path")
    if not image_value:
        return None
    image_path = Path(str(image_value))
    if image_root and not image_path.is_absolute():
        image_path = image_root / image_path.name
    return image_path


def audit_records(records: list[dict[str, Any]], image_root: Path | None = None) -> dict[str, Any]:
    categories: Counter[str] = Counter()
    subcategories: Counter[str] = Counter()
    capabilities: Counter[str] = Counter()
    shooting_angles: Counter[str] = Counter()
    weather: Counter[str] = Counter()
    missing_images = 0
    usable_records = 0

    for record in records:
        reference = normalize_reference(record.get("reference") or record.get("answer"))
        capability = infer_external_vqa_capability(
            instruction=str(record.get("question") or record.get("instruction") or ""),
            category=str(record.get("category") or ""),
            subcategory=str(record.get("subcategory") or record.get("sub_category") or ""),
            reference=reference,
        )
        capabilities[capability] += 1
        categories[str(record.get("category") or "unknown")] += 1
        subcategories[str(record.get("subcategory") or record.get("sub_category") or "unknown")] += 1
        shooting_angles[str(record.get("shooting_angle") or "unknown")] += 1
        weather[str(record.get("weather_conditions") or record.get("weather") or "unknown")] += 1

        image_path = resolve_image_path(record, image_root)
        if image_path is None or not image_path.exists():
            missing_images += 1
        else:
            usable_records += 1

    return {
        "total_records": len(records),
        "usable_records_with_images": usable_records,
        "missing_images": missing_images,
        "capability_counts": dict(capabilities),
        "category_counts": dict(categories),
        "subcategory_counts": dict(subcategories),
        "shooting_angle_counts": dict(shooting_angles),
        "weather_counts": dict(weather),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit IntelliCockpitBench JSONL metadata and image availability.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--image_root", default="")
    parser.add_argument("--output", default="outputs/eval_results/intelli_dataset_audit.json")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    image_root = Path(args.image_root) if args.image_root else None
    report = audit_records(records, image_root)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote audit report to {args.output}")


if __name__ == "__main__":
    main()
