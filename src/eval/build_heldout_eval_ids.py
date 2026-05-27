"""Build reproducible held-out ID lists not seen in SFT-v3-r3 training."""
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

from src.eval.audit_train_eval_overlap import SETTINGS, normalize_id, normalized_ids, read_jsonl


def available_heldout_ids(train_file: Path, visual_control_dir: Path, dataset: str) -> tuple[list[str], dict[str, int]]:
    train_ids = normalized_ids(read_jsonl(train_file))
    maps: dict[str, set[str]] = {}
    for setting in SETTINGS:
        path = visual_control_dir / f"{dataset}_strict_{setting}.jsonl"
        maps[setting] = {str(row.get("id")) for row in read_jsonl(path) if row.get("id")}
    common_ids = set.intersection(*maps.values())
    heldout = sorted(sample_id for sample_id in common_ids if normalize_id(sample_id) not in train_ids)
    counts = {"train_unique_ids": len(train_ids), "aligned_visual_control_ids": len(common_ids), "heldout_available": len(heldout)}
    return heldout, counts


def build_payload(candidates: list[str], requested: int, seed: int, counts: dict[str, int]) -> dict[str, Any]:
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    chosen = shuffled[:requested]
    warnings: list[str] = []
    if len(chosen) < requested:
        warnings.append(f"requested {requested} held-out ids, but only {len(chosen)} are available")
    if not chosen:
        warnings.append("no held-out ids are available in the current strict visual-control pool; do not run held-out eval")
    return {
        "ids": chosen,
        "requested": requested,
        "selected": len(chosen),
        "seed": seed,
        **counts,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select held-out strict visual-control IDs.")
    parser.add_argument("--train_file", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    parser.add_argument("--visual_control_dir", default="data/processed/visual_control")
    parser.add_argument("--dataset", default="lingoqa")
    parser.add_argument("--output_dir", default="outputs/final_report")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates, counts = available_heldout_ids(Path(args.train_file), Path(args.visual_control_dir), args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for size in (100, 200):
        payload = build_payload(candidates, size, args.seed, counts)
        path = output_dir / f"stage4_5_heldout_ids_{size}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payloads[str(size)] = {"path": path.as_posix(), "selected": payload["selected"], "warnings": payload["warnings"]}
    print(json.dumps({"counts": counts, "outputs": payloads}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
