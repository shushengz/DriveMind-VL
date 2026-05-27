"""Build model-mined answer-only Preference-v8.1 pairs from r3 hard negatives."""
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
TARGETS = {
    "mined_text_prior_bias": 0.22,
    "mined_blank_prior_answer": 0.18,
    "mined_wrong_image_confound": 0.17,
    "control_over_gold_overlap": 0.10,
    "normal_anchor_gold_vs_model_wrong": 0.26,
    "spatial_gold_vs_spatial_wrong": 0.07,
}
HARD_TO_PAIR = {
    "text_prior_bias": "mined_text_prior_bias",
    "blank_prior_answer": "mined_blank_prior_answer",
    "wrong_image_confound": "mined_wrong_image_confound",
    "control_over_gold_overlap": "control_over_gold_overlap",
    "normal_wrong_anchor": "normal_anchor_gold_vs_model_wrong",
    "spatial_relation_error": "spatial_gold_vs_spatial_wrong",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_hard_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"hard-negative file does not exist; run r3 mining first: {path}")
    return read_jsonl(path)


def load_visual_rows(directory: Path) -> dict[str, dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {}
    for setting in ("normal", "text_only", "wrong_image", "blank_image"):
        maps[setting] = {
            str(row.get("id")): row
            for row in read_jsonl(directory / f"lingoqa_strict_{setting}.jsonl")
            if row.get("id")
        }
    return maps


def quotas(max_pairs: int) -> dict[str, int]:
    values = {key: int(max_pairs * ratio) for key, ratio in TARGETS.items()}
    remainder = max_pairs - sum(values.values())
    for key in TARGETS:
        if remainder <= 0:
            break
        values[key] += 1
        remainder -= 1
    return values


def selected_answer(hard: dict[str, Any], setting: str) -> str:
    return str(hard.get(f"{setting}_answer", ""))


def pair_from_hard(hard: dict[str, Any], pair_type: str, visual: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    source_id = str(hard["source_id"])
    setting = str(hard.get("setting") or "normal")
    source = visual.get(setting, {}).get(source_id) or visual["normal"].get(source_id)
    if source is None:
        raise ValueError(f"missing visual-control source row for {source_id} ({setting})")
    if pair_type in {"mined_text_prior_bias", "mined_blank_prior_answer"}:
        chosen = NEUTRAL_ABSTAIN
    elif pair_type in {"mined_wrong_image_confound", "control_over_gold_overlap"}:
        chosen = NEUTRAL_CAUTION if setting == "wrong_image" else NEUTRAL_ABSTAIN
    else:
        chosen = str(hard.get("gold", ""))
        setting = "normal"
        source = visual["normal"][source_id]
    rejected_setting = setting if pair_type not in {"normal_anchor_gold_vs_model_wrong", "spatial_gold_vs_spatial_wrong"} else "normal"
    rejected = selected_answer(hard, rejected_setting)
    return {
        "id": f"{source_id}_{pair_type}",
        "dataset": "lingoqa",
        "source_id": source_id,
        "pair_type": pair_type,
        "hard_type": str(hard["hard_type"]),
        "setting": setting,
        "prompt": str(source.get("prompt", "")),
        "image_paths": list(source.get("image_paths") or []),
        "image_labels": list(source.get("image_labels") or []),
        "chosen": {"answer": chosen},
        "rejected": {"answer": rejected},
        "weight": 1.0,
        "metadata": {
            "source_model": "sft_v3_r3_lingo_smoke",
            "source_original_id": str((source.get("metadata") or {}).get("original_id", "")),
            "hard_type": str(hard["hard_type"]),
            "trigger_reason": str(hard.get("trigger_reason", "")),
            "rejected_from_r3_actual_prediction": True,
            "heldout_excluded": True,
        },
    }


def build_pairs(
    hard_rows: list[dict[str, Any]],
    visual: dict[str, dict[str, dict[str, Any]]],
    heldout_keys: set[str],
    max_pairs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = {pair_type: [] for pair_type in TARGETS}
    for hard in hard_rows:
        if identity_keys(hard.get("source_id")) & heldout_keys:
            raise ValueError(f"held-out sample present in hard negatives: {hard.get('source_id')}")
        pair_type = HARD_TO_PAIR.get(str(hard.get("hard_type")))
        if pair_type:
            grouped[pair_type].append(hard)
    rng = random.Random(seed)
    wanted = quotas(max_pairs)
    pairs: list[dict[str, Any]] = []
    available = {key: len(rows) for key, rows in grouped.items()}
    for pair_type, rows in grouped.items():
        shuffled = list(rows)
        rng.shuffle(shuffled)
        for hard in shuffled[: wanted[pair_type]]:
            pairs.append(pair_from_hard(hard, pair_type, visual))
    rng.shuffle(pairs)
    return pairs, available


def stats(pairs: list[dict[str, Any]], available: dict[str, int]) -> dict[str, Any]:
    by_type = Counter(str(pair["pair_type"]) for pair in pairs)
    by_hard = Counter(str(pair["hard_type"]) for pair in pairs)
    normal = by_type["normal_anchor_gold_vs_model_wrong"]
    spatial = by_type["spatial_gold_vs_spatial_wrong"]
    control = sum(by_type[key] for key in (
        "mined_text_prior_bias",
        "mined_blank_prior_answer",
        "mined_wrong_image_confound",
        "control_over_gold_overlap",
    ))
    total = len(pairs)
    return {
        "total_pairs": total,
        "uses_model_predictions": True,
        "rejected_from_r3_actual_predictions": True,
        "pair_type_counts": dict(by_type),
        "pair_type_ratios": {key: count / total if total else 0.0 for key, count in by_type.items()},
        "hard_type_counts": dict(by_hard),
        "available_pair_type_counts": available,
        "normal_pair_ratio": normal / total if total else 0.0,
        "control_pair_ratio": control / total if total else 0.0,
        "spatial_pair_ratio": spatial / total if total else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-mined Preference-v8.1 pairs.")
    parser.add_argument("--hard_negatives", default="data/mining/v8_1/hard_negatives_v8_1.jsonl")
    parser.add_argument("--sft_v3_data", "--sft_r3_data", dest="sft_v3_data", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl", help="Accepted as the r3 anchor provenance input.")
    parser.add_argument("--visual_control_dir", default="data/processed/visual_control")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_dir", default="data/train/preference_v8_1")
    parser.add_argument("--max_pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true", help="Accepted for orchestration compatibility; construction is offline only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hard_rows = read_hard_rows(Path(args.hard_negatives))
    _, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    visual = load_visual_rows(Path(args.visual_control_dir))
    pairs, available = build_pairs(hard_rows, visual, heldout_keys, args.max_pairs, args.seed)
    if not pairs:
        raise SystemExit("no model-mined pairs could be constructed")
    output = Path(args.output_dir)
    write_jsonl(output / "preference_v8_1_pairs.jsonl", pairs)
    summary = stats(pairs, available)
    write_json(output / "preference_v8_1_stats.json", summary)
    preview = []
    for pair_type in TARGETS:
        preview.extend([pair for pair in pairs if pair["pair_type"] == pair_type][:5])
    write_jsonl(output / "preference_v8_1_preview.jsonl", preview)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
