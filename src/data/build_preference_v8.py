"""Build answer-only, held-out-excluded DPO-v8 preference pairs.

This Stage 5 builder is CPU-only and rule-based when no approved train-pool
prediction source is supplied. Held-out evaluation rows are never pair inputs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.exclude_heldout_ids import filter_candidates, identity_keys, read_heldout_keys, read_jsonl, write_jsonl

PAIR_TARGETS = {
    "normal_anchor_gold_vs_bad": 0.42,
    "normal_gold_vs_refusal": 0.13,
    "control_abstain_vs_hallucination": 0.22,
    "wrong_image_caution_vs_confident_answer": 0.15,
    "spatial_gold_vs_spatial_wrong": 0.08,
}
SPATIAL_RE = re.compile(r"\bleft\b|\bright\b|\bfront\b|\bback\b|\blane\b|traffic light|pedestrian|vehicle|behind|ahead", re.I)
NEUTRAL_ABSTAIN = "Insufficient evidence to answer reliably."
NEUTRAL_CAUTION = "The provided input does not support a reliable answer."


def load_settings(visual_control_dir: Path) -> dict[str, list[dict[str, Any]]]:
    return {setting: read_jsonl(visual_control_dir / f"lingoqa_strict_{setting}.jsonl") for setting in ("normal", "text_only", "wrong_image", "blank_image")}


def by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows if row.get("id")}


def answer_only(text: str) -> dict[str, str]:
    return {"answer": str(text or "")}


def incorrect_answer(gold: str) -> str:
    norm = gold.strip().lower().rstrip(".")
    if norm in {"no", "no,", "false"} or norm.startswith("no "):
        return "Yes."
    if norm in {"yes", "true"} or norm.startswith("yes"):
        return "No."
    return NEUTRAL_ABSTAIN


def spatial_wrong(gold: str) -> str:
    swaps = [("left", "right"), ("right", "left"), ("front", "behind"), ("behind", "front"), ("ahead", "behind")]
    lower = gold.lower()
    for old, new in swaps:
        if old in lower:
            return re.sub(old, new, gold, count=1, flags=re.I)
    return "The relevant object is on the opposite side."


def make_pair(row: dict[str, Any], pair_type: str, setting: str, chosen: str, rejected: str, suffix: int = 0) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    source_id = str(row.get("id") or "")
    source_original_id = str(metadata.get("original_id") or "")
    pair_id = f"{source_id}_{pair_type}" + (f"_{suffix}" if suffix else "")
    return {
        "id": pair_id,
        "dataset": "lingoqa",
        "source_id": source_id,
        "pair_type": pair_type,
        "setting": setting,
        "prompt": str(row.get("prompt") or ""),
        "image_paths": list(row.get("image_paths") or []),
        "image_labels": list(row.get("image_labels") or []),
        "chosen": answer_only(chosen),
        "rejected": answer_only(rejected),
        "weight": 1.0,
        "metadata": {
            "source_model": "sft_v3_r3_lingo_smoke",
            "source_original_id": source_original_id,
            "difficulty": str(metadata.get("difficulty") or "unknown"),
            "f1_gap": None,
            "hallucination_trigger": "rule_based_unsupported_confident_answer",
            "heldout_excluded": True,
        },
    }


def allocated_counts(total: int) -> dict[str, int]:
    counts = {key: int(total * ratio) for key, ratio in PAIR_TARGETS.items()}
    remaining = total - sum(counts.values())
    for key in PAIR_TARGETS:
        if remaining <= 0:
            break
        counts[key] += 1
        remaining -= 1
    return counts


def take(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = list(rows)
    rng.shuffle(candidates)
    return candidates[: min(count, len(candidates))]


def build_pairs(
    rows_by_setting: dict[str, list[dict[str, Any]]],
    heldout_keys: set[str],
    max_pairs: int,
    seed: int,
    uses_model_predictions: bool = False,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    normal, removed = filter_candidates(rows_by_setting["normal"], heldout_keys)
    allowed_ids = {str(row.get("id")) for row in normal}
    controls = {setting: [row for row in rows_by_setting[setting] if str(row.get("id")) in allowed_ids] for setting in ("text_only", "wrong_image", "blank_image")}
    total = max_pairs or 500
    targets = allocated_counts(total)
    spatial = [row for row in normal if SPATIAL_RE.search(str(row.get("question") or ""))]
    pairs: list[dict[str, Any]] = []

    for row in take(normal, targets["normal_anchor_gold_vs_bad"], rng):
        gold = str(row.get("gold") or "")
        pairs.append(make_pair(row, "normal_anchor_gold_vs_bad", "normal", gold, incorrect_answer(gold)))
    for row in take(normal, targets["normal_gold_vs_refusal"], rng):
        pairs.append(make_pair(row, "normal_gold_vs_refusal", "normal", str(row.get("gold") or ""), NEUTRAL_ABSTAIN))

    mixed_control = controls["text_only"] + controls["blank_image"] + controls["wrong_image"]
    for row in take(mixed_control, targets["control_abstain_vs_hallucination"], rng):
        setting = str(row.get("setting") or "text_only")
        chosen = NEUTRAL_CAUTION if setting == "wrong_image" else NEUTRAL_ABSTAIN
        pairs.append(make_pair(row, "control_abstain_vs_hallucination", setting, chosen, str(row.get("gold") or "")))
    for row in take(controls["wrong_image"], targets["wrong_image_caution_vs_confident_answer"], rng):
        pairs.append(make_pair(row, "wrong_image_caution_vs_confident_answer", "wrong_image", NEUTRAL_CAUTION, str(row.get("gold") or "")))
    for row in take(spatial, targets["spatial_gold_vs_spatial_wrong"], rng):
        gold = str(row.get("gold") or "")
        pairs.append(make_pair(row, "spatial_gold_vs_spatial_wrong", "normal", gold, spatial_wrong(gold)))

    if len(pairs) < total:
        existing = {(pair["source_id"], pair["pair_type"]) for pair in pairs}
        for row in normal:
            key = (str(row.get("id")), "normal_anchor_gold_vs_bad")
            if key in existing:
                continue
            gold = str(row.get("gold") or "")
            pairs.append(make_pair(row, "normal_anchor_gold_vs_bad", "normal", gold, incorrect_answer(gold)))
            if len(pairs) >= total:
                break
    rng.shuffle(pairs)
    return pairs[:total]


def stats(pairs: list[dict[str, Any]], heldout_ids: list[str], uses_model_predictions: bool) -> dict[str, Any]:
    by_type = Counter(str(pair.get("pair_type")) for pair in pairs)
    by_setting = Counter(str(pair.get("setting")) for pair in pairs)
    normal_types = {"normal_anchor_gold_vs_bad", "normal_gold_vs_refusal"}
    control_types = {"control_abstain_vs_hallucination", "wrong_image_caution_vs_confident_answer"}
    normal_count = sum(by_type[pair_type] for pair_type in normal_types)
    control_count = sum(by_type[pair_type] for pair_type in control_types)
    spatial_count = by_type["spatial_gold_vs_spatial_wrong"]
    total = len(pairs)
    return {
        "total_pairs": total,
        "pair_type_counts": dict(by_type),
        "pair_type_ratios": {key: value / total if total else 0.0 for key, value in by_type.items()},
        "setting_counts": dict(by_setting),
        "normal_pair_ratio": normal_count / total if total else 0.0,
        "control_pair_ratio": control_count / total if total else 0.0,
        "spatial_pair_ratio": spatial_count / total if total else 0.0,
        "weight_sum": sum(float(pair.get("weight", 0.0)) for pair in pairs),
        "uses_model_predictions": bool(uses_model_predictions),
        "heldout_id_count_used_for_exclusion_only": len(heldout_ids),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build answer-only DPO-v8 pairs from the permitted training pool.")
    parser.add_argument("--visual_control_dir", default="data/processed/visual_control")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_dir", default="data/train/preference_v8")
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--sft_v3_data", default="", help="Accepted for backward CLI compatibility; unused by Stage 5.")
    parser.add_argument("--case_scores", default="", help="Accepted for backward CLI compatibility; held-out cases are never read.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_ids, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    rows_by_setting = load_settings(Path(args.visual_control_dir))
    max_pairs = min(args.max_pairs, 50) if args.dry_run else args.max_pairs
    pairs = build_pairs(rows_by_setting, heldout_keys, max_pairs=max_pairs, seed=args.seed, uses_model_predictions=False)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "preference_v8_pairs.jsonl", pairs)
    summary = stats(pairs, heldout_ids, uses_model_predictions=False)
    write_json(output_dir / "preference_v8_stats.json", summary)
    preview = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        grouped.setdefault(str(pair["pair_type"]), []).append(pair)
    for pair_type in PAIR_TARGETS:
        preview.extend(grouped.get(pair_type, [])[:5])
    write_jsonl(output_dir / "preference_v8_preview.jsonl", preview)
    print(json.dumps({"output_dir": output_dir.as_posix(), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
