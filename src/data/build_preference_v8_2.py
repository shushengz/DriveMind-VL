"""Build case-gap-aware Preference-v8.2 pairs without model execution."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.exclude_heldout_ids import identity_keys, read_heldout_keys, read_jsonl, write_jsonl

NEUTRAL_ABSTAIN = "Insufficient evidence to answer reliably."
NEUTRAL_CAUTION = "The provided input does not support a reliable answer."
TARGET_RATIOS = {
    "normal_anchor_gold_vs_model_wrong": 0.35,
    "normal_gold_vs_refusal": 0.05,
    "control_direct_answer_vs_caution": 0.20,
    "blank_high_f1_vs_caution": 0.15,
    "text_only_gold_overlap_vs_caution": 0.10,
    "wrong_image_gold_overlap_vs_caution": 0.10,
    "spatial_gold_vs_spatial_wrong": 0.05,
}
PAIR_TO_CANDIDATE = {
    "normal_anchor_gold_vs_model_wrong": ["normal_model_wrong", "normal_gold_vs_refusal_anchor"],
    "normal_gold_vs_refusal": ["normal_gold_vs_refusal_anchor"],
    "control_direct_answer_vs_caution": ["control_direct_answer", "control_over_gold_overlap"],
    "blank_high_f1_vs_caution": ["blank_high_f1_prior"],
    "text_only_gold_overlap_vs_caution": ["text_only_gold_overlap"],
    "wrong_image_gold_overlap_vs_caution": ["wrong_image_gold_overlap"],
    "spatial_gold_vs_spatial_wrong": ["spatial_relation_error"],
}
CONTROL_TYPES = {
    "control_direct_answer_vs_caution",
    "blank_high_f1_vs_caution",
    "text_only_gold_overlap_vs_caution",
    "wrong_image_gold_overlap_vs_caution",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def quotas(max_pairs: int) -> dict[str, int]:
    output = {key: int(max_pairs * ratio) for key, ratio in TARGET_RATIOS.items()}
    remainder = max_pairs - sum(output.values())
    for key in TARGET_RATIOS:
        if remainder <= 0:
            break
        output[key] += 1
        remainder -= 1
    return output


def load_visual_rows(directory: Path) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        setting: {str(row.get("id")): row for row in read_jsonl(directory / f"lingoqa_strict_{setting}.jsonl")}
        for setting in ("normal", "text_only", "wrong_image", "blank_image")
    }


def choose_answer(pair_type: str, candidate: dict[str, Any]) -> tuple[str, str]:
    if pair_type in {"normal_anchor_gold_vs_model_wrong", "normal_gold_vs_refusal", "spatial_gold_vs_spatial_wrong"}:
        return str(candidate.get("gold", "")), (
            NEUTRAL_ABSTAIN if pair_type == "normal_gold_vs_refusal" else str(candidate.get("model_answer", ""))
        )
    if pair_type == "wrong_image_gold_overlap_vs_caution":
        return NEUTRAL_CAUTION, str(candidate.get("model_answer", ""))
    return NEUTRAL_ABSTAIN, str(candidate.get("model_answer", ""))


def make_pair(pair_type: str, candidate: dict[str, Any], visual: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any] | None:
    source_id = str(candidate.get("source_id", ""))
    setting = "normal" if pair_type.startswith("normal_") or pair_type.startswith("spatial_") else str(candidate.get("setting", ""))
    source = visual.get(setting, {}).get(source_id)
    if source is None:
        return None
    chosen, rejected = choose_answer(pair_type, candidate)
    if not chosen or not rejected or chosen.strip() == rejected.strip():
        return None
    actual = bool(candidate.get("rejected_from_actual_prediction")) and pair_type != "normal_gold_vs_refusal"
    if pair_type == "normal_anchor_gold_vs_model_wrong" and candidate.get("candidate_type") == "normal_gold_vs_refusal_anchor":
        actual = False
    return {
        "id": f"{source_id}_{pair_type}",
        "dataset": "lingoqa",
        "source_id": source_id,
        "pair_type": pair_type,
        "setting": setting,
        "prompt": str(source.get("prompt", "")),
        "image_paths": list(source.get("image_paths") or []),
        "image_labels": list(source.get("image_labels") or []),
        "chosen": {"answer": chosen},
        "rejected": {"answer": rejected},
        "weight": 1.0,
        "metadata": {
            "source_model": "sft_v3_r3_or_dpo_v8_1",
            "source_stage": "stage8_5",
            "candidate_type": str(candidate.get("candidate_type", "")),
            "control_f1": float(candidate.get("control_f1", 0.0)),
            "case_gap": float(candidate.get("case_gap", 0.0)),
            "heldout_excluded": True,
            "source_file": str(candidate.get("source_file", "")),
            "rejected_from_actual_prediction": actual,
            "source_original_id": str((source.get("metadata") or {}).get("original_id", "")),
        },
    }


def build_pairs(candidates: list[dict[str, Any]], visual: dict[str, dict[str, dict[str, Any]]], heldout_keys: set[str], max_pairs: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    leaking = [row.get("source_id") for row in candidates if identity_keys(row.get("source_id")) & heldout_keys]
    if leaking:
        raise ValueError(f"held-out IDs found in v8.2 candidate source: {leaking[:5]}")
    rng = random.Random(seed)
    pools: dict[str, list[dict[str, Any]]] = {}
    for candidate_type in {item for items in PAIR_TO_CANDIDATE.values() for item in items}:
        rows = [row for row in candidates if row.get("candidate_type") == candidate_type]
        rng.shuffle(rows)
        pools[candidate_type] = rows
    targets = quotas(max_pairs)
    pairs: list[dict[str, Any]] = []
    used_signatures: set[tuple[str, str, str, str]] = set()
    warnings: list[str] = []

    def take(pair_type: str, wanted: int) -> int:
        added = 0
        for candidate_type in PAIR_TO_CANDIDATE[pair_type]:
            for candidate in pools[candidate_type]:
                if added >= wanted:
                    return added
                pair = make_pair(pair_type, candidate, visual)
                if pair is None:
                    continue
                signature = (pair["source_id"], pair["setting"], pair["chosen"]["answer"], pair["rejected"]["answer"])
                if signature in used_signatures:
                    continue
                used_signatures.add(signature)
                pairs.append(pair)
                added += 1
        return added

    # Reserve specialized high-overlap controls before the broader direct pool.
    order = [
        "blank_high_f1_vs_caution",
        "text_only_gold_overlap_vs_caution",
        "wrong_image_gold_overlap_vs_caution",
        # Spatial errors are a narrow subset of normal failures; reserve them
        # before the broader normal-anchor pool consumes identical signatures.
        "spatial_gold_vs_spatial_wrong",
        "normal_anchor_gold_vs_model_wrong",
        "normal_gold_vs_refusal",
        "control_direct_answer_vs_caution",
    ]
    shortages: dict[str, int] = {}
    for pair_type in order:
        made = take(pair_type, targets[pair_type])
        if made < targets[pair_type]:
            shortages[pair_type] = targets[pair_type] - made
            warnings.append(f"{pair_type} has {made} usable pairs below requested {targets[pair_type]}; no duplicates or held-out rows were added.")
    missing = max_pairs - len(pairs)
    if missing:
        # Reallocate control shortages to direct-answer controls and other
        # shortages to normal anchors, preserving the 40/55/5 design where possible.
        direct_missing = sum(value for key, value in shortages.items() if key in CONTROL_TYPES)
        extra_direct = take("control_direct_answer_vs_caution", min(missing, direct_missing))
        missing -= extra_direct
        if missing:
            missing -= take("normal_anchor_gold_vs_model_wrong", missing)
        if missing:
            warnings.append(f"Could not fill {missing} of {max_pairs} requested pairs without reuse.")
    rng.shuffle(pairs)
    available = {key: sum(len(pools[item]) for item in candidate_types) for key, candidate_types in PAIR_TO_CANDIDATE.items()}
    return pairs[:max_pairs], {"requested_targets": targets, "available_candidates": available, "warnings": warnings}


def build_stats(pairs: list[dict[str, Any]], build_info: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(pair["pair_type"] for pair in pairs)
    total = len(pairs)
    normal = sum(counts[key] for key in ("normal_anchor_gold_vs_model_wrong", "normal_gold_vs_refusal"))
    spatial = counts["spatial_gold_vs_spatial_wrong"]
    control = sum(counts[key] for key in CONTROL_TYPES)
    return {
        "total_pairs": total,
        "uses_model_predictions": True,
        "pair_type_counts": dict(counts),
        "pair_type_ratios": {key: count / total if total else 0.0 for key, count in sorted(counts.items())},
        "normal_pair_ratio": normal / total if total else 0.0,
        "control_pair_ratio": control / total if total else 0.0,
        "spatial_pair_ratio": spatial / total if total else 0.0,
        "rejected_from_actual_prediction_rate": sum(bool((pair.get("metadata") or {}).get("rejected_from_actual_prediction")) for pair in pairs) / total if total else 0.0,
        "requested_targets": build_info["requested_targets"],
        "available_candidates": build_info["available_candidates"],
        "warnings": build_info["warnings"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build case-gap-aware Preference-v8.2 pairs.")
    parser.add_argument("--candidates", default="data/mining/v8_2/preference_v8_2_candidate_cases.jsonl")
    parser.add_argument("--sft_r3_data", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    parser.add_argument("--visual_control_dir", default="data/processed/visual_control")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_dir", default="data/train/preference_v8_2")
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_jsonl(Path(args.candidates))
    if args.dry_run:
        args.max_pairs = min(args.max_pairs, 50)
    _, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    visual = load_visual_rows(Path(args.visual_control_dir))
    pairs, info = build_pairs(candidates, visual, heldout_keys, args.max_pairs, args.seed)
    if not pairs:
        raise SystemExit("no Preference-v8.2 pairs could be built")
    stats = build_stats(pairs, info)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "preference_v8_2_pairs.jsonl", pairs)
    write_json(output_dir / "preference_v8_2_stats.json", stats)
    preview: list[dict[str, Any]] = []
    for pair_type in TARGET_RATIOS:
        preview.extend([pair for pair in pairs if pair["pair_type"] == pair_type][:5])
    write_jsonl(output_dir / "preference_v8_2_preview.jsonl", preview)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
