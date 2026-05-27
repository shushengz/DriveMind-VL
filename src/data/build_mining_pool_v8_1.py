"""Build a deterministic non-held-out LingoQA mining pool for Preference-v8.1."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.exclude_heldout_ids import identity_keys, read_heldout_keys, read_jsonl

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_pool(
    visual_control_dir: Path,
    heldout_ids_path: Path,
    max_ids: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    heldout_ids, heldout_keys = read_heldout_keys(heldout_ids_path)
    maps = {}
    for setting in SETTINGS:
        rows = read_jsonl(visual_control_dir / f"lingoqa_strict_{setting}.jsonl")
        maps[setting] = {str(row.get("id")): row for row in rows if row.get("id")}
    common = set.intersection(*(set(maps[setting]) for setting in SETTINGS))
    ordered = [sample_id for sample_id in maps["normal"] if sample_id in common]
    removed = []
    eligible = []
    for sample_id in ordered:
        row = maps["normal"][sample_id]
        if identity_keys(sample_id, row) & heldout_keys:
            removed.append(sample_id)
        else:
            eligible.append(sample_id)
    rng = random.Random(seed)
    selected = list(eligible)
    rng.shuffle(selected)
    selected = selected[:max_ids] if max_ids else selected
    leaking = [
        sample_id for sample_id in selected
        if identity_keys(sample_id, maps["normal"][sample_id]) & heldout_keys
    ]
    warnings = []
    if max_ids and len(eligible) < max_ids:
        warnings.append(f"requested {max_ids} mining IDs but only {len(eligible)} are eligible")
    payload = {
        "dataset": "lingoqa",
        "purpose": "model_mined_preference_v8_1",
        "seed": seed,
        "ids": selected,
    }
    summary = {
        "total_candidates": len(common),
        "heldout_id_count": len(heldout_ids),
        "heldout_removed": len(removed),
        "heldout_removed_ids": removed,
        "eligible_after_filter": len(eligible),
        "final_mining_ids": len(selected),
        "leakage_after_filter": len(leaking),
        "leakage_after_filter_ids": leaking,
        "all_settings_complete": True,
        "warnings": warnings,
        "train_ready": len(leaking) == 0 and bool(selected),
    }
    return payload, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the non-held-out r3 mining pool.")
    parser.add_argument("--visual_control_dir", default="data/processed/visual_control")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_ids", default="data/mining/v8_1/lingoqa_mining_pool_ids.json")
    parser.add_argument("--output_summary", default="data/mining/v8_1/lingoqa_mining_pool_summary.json")
    parser.add_argument("--max_ids", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_ids = min(args.max_ids, 16) if args.dry_run else args.max_ids
    payload, summary = build_pool(Path(args.visual_control_dir), Path(args.heldout_ids), max_ids, args.seed)
    write_json(Path(args.output_ids), payload)
    write_json(Path(args.output_summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["train_ready"]:
        raise SystemExit("mining pool contains held-out leakage or no eligible samples")


if __name__ == "__main__":
    main()
