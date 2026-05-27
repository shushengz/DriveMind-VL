"""Build v7 preference data with stricter visual-contrast mining.

v5/v6 showed a recurring failure mode: control refusal pairs can suppress the
model's normal visual answering ability, especially on spatial localization.
This builder only mines control pairs when the normal prediction is strong and
the ablated-input prediction is clearly worse, then compensates with stronger
normal SFT anchors for fragile capabilities.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.build_lingoqa_preference_v5_from_predictions import (  # noqa: E402
    MODE_WEIGHTS,
    answer_obj,
    answer_text,
    capability,
    index_by_id,
    make_pair,
    prediction_obj,
    read_jsonl,
    refusal_for,
    write_jsonl,
)
from src.eval.refusal_detection import is_refusal_prediction  # noqa: E402
from src.eval.run_all_eval import token_f1  # noqa: E402


def parse_weight_map(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid weight map item: {item}")
        key, value = item.split("=", 1)
        weights[key.strip()] = float(value)
    return weights


def cap_anchor_weight(cap: str, args: argparse.Namespace) -> float:
    weights = parse_weight_map(args.capability_sft_anchor_weights)
    return weights.get(cap, args.normal_sft_anchor_weight)


def cap_control_weight_multiplier(cap: str, args: argparse.Namespace) -> float:
    weights = parse_weight_map(args.capability_control_weight_multipliers)
    return weights.get(cap, 1.0)


def normal_prediction_f1(
    sample_id: str,
    gold: dict[str, Any],
    normal_predictions: dict[str, dict[str, Any]] | None,
) -> float | None:
    if normal_predictions is None:
        return None
    pred_row = normal_predictions.get(sample_id)
    if pred_row is None:
        return None
    pred = prediction_obj(pred_row.get("prediction"))
    return token_f1(answer_text(pred), answer_text(gold))


def add_normal_pairs(
    pairs: list[dict[str, Any]],
    normal_rows: list[dict[str, Any]],
    normal_predictions: dict[str, dict[str, Any]] | None,
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    for row in normal_rows:
        gold = answer_obj(row)
        if not gold["answer"]:
            counters["normal_skipped_empty_gold"] += 1
            continue
        pred_row = normal_predictions.get(str(row.get("id"))) if normal_predictions else None
        rejected = refusal_for("normal")
        pair_type = "normal_answer_over_refusal"
        meta_extra = {"rejected_source": "template_refusal"}
        if pred_row is not None:
            pred = prediction_obj(pred_row.get("prediction"))
            pred_f1 = token_f1(answer_text(pred), answer_text(gold))
            if is_refusal_prediction(pred_row.get("prediction")) or pred_f1 <= args.normal_bad_f1_threshold:
                rejected = pred
                pair_type = "normal_answer_over_bad_prediction"
                meta_extra = {
                    "rejected_source": "model_prediction",
                    "normal_prediction_f1_vs_gold": round(pred_f1, 4),
                }

        cap = capability(row)
        pairs.append(
            make_pair(
                row=row,
                chosen=gold,
                rejected=rejected,
                pair_type=pair_type,
                mode="normal",
                weight=args.normal_weight,
                sft_anchor_weight=cap_anchor_weight(cap, args),
                meta_extra=meta_extra,
            )
        )


def maybe_add_control_pair(
    pairs: list[dict[str, Any]],
    mode: str,
    sample_id: str,
    source_row: dict[str, Any],
    gold_row: dict[str, Any],
    pred_row: dict[str, Any],
    normal_predictions: dict[str, dict[str, Any]] | None,
    args: argparse.Namespace,
    counters: Counter[str],
    f1_sums: dict[str, float],
    by_capability_added: Counter[str],
) -> None:
    gold = answer_obj(gold_row)
    cap = capability(gold_row)
    counters[f"{mode}_seen"] += 1
    normal_f1 = normal_prediction_f1(sample_id, gold, normal_predictions)
    if normal_f1 is None:
        counters[f"{mode}_skipped_missing_normal_prediction"] += 1
        return
    if normal_f1 < args.min_normal_prediction_f1_for_control:
        counters[f"{mode}_skipped_low_normal_prediction_f1"] += 1
        return

    rejected = prediction_obj(pred_row.get("prediction"))
    pred_text = answer_text(rejected)
    control_f1 = token_f1(pred_text, answer_text(gold))
    f1_sums[mode] += control_f1
    if is_refusal_prediction(pred_row.get("prediction")):
        counters[f"{mode}_skipped_already_refusal"] += 1
        return
    if not pred_text.strip():
        counters[f"{mode}_skipped_empty_prediction"] += 1
        return
    if control_f1 > args.max_control_prediction_f1:
        counters[f"{mode}_skipped_high_control_f1"] += 1
        return
    if normal_f1 - control_f1 < args.min_visual_contrast_gap:
        counters[f"{mode}_skipped_low_visual_contrast_gap"] += 1
        return
    if args.max_control_pairs_per_capability > 0 and by_capability_added[cap] >= args.max_control_pairs_per_capability:
        counters[f"{mode}_skipped_capability_quota"] += 1
        return

    weight = float(getattr(args, f"{mode}_weight")) * cap_control_weight_multiplier(cap, args)
    pairs.append(
        make_pair(
            row=source_row,
            chosen=refusal_for(mode),
            rejected=rejected,
            pair_type="control_refusal_over_low_quality_ablated_prediction",
            mode=mode,
            weight=weight,
            sft_anchor_weight=0.0,
            meta_extra={
                "rejected_source": "sft_v2_prediction",
                "prediction_f1_vs_gold": round(control_f1, 4),
                "normal_prediction_f1_vs_gold": round(normal_f1, 4),
                "visual_contrast_gap": round(normal_f1 - control_f1, 4),
                "prediction_was_refusal": False,
            },
        )
    )
    counters[f"{mode}_added"] += 1
    by_capability_added[cap] += 1


def add_control_pairs(
    pairs: list[dict[str, Any]],
    mode: str,
    source_by_id: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    gold_by_id: dict[str, dict[str, Any]],
    normal_predictions: dict[str, dict[str, Any]] | None,
    args: argparse.Namespace,
    counters: Counter[str],
    f1_sums: dict[str, float],
    by_capability_added: Counter[str],
) -> None:
    added = 0
    max_pairs = int(getattr(args, f"max_{mode}_pairs"))
    for sample_id, pred_row in predictions.items():
        if max_pairs > 0 and added >= max_pairs:
            counters[f"{mode}_skipped_max_pairs"] += 1
            continue
        source_row = source_by_id.get(sample_id)
        gold_row = gold_by_id.get(sample_id)
        if source_row is None or gold_row is None:
            counters[f"{mode}_skipped_missing_source"] += 1
            continue
        before = counters[f"{mode}_added"]
        maybe_add_control_pair(
            pairs,
            mode,
            sample_id,
            source_row,
            gold_row,
            pred_row,
            normal_predictions,
            args,
            counters,
            f1_sums,
            by_capability_added,
        )
        if counters[f"{mode}_added"] > before:
            added += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v7 grounded preference pairs from SFT-v2 train predictions.")
    parser.add_argument("--normal", default="data/processed/lingoqa_clean_v2_train.jsonl")
    parser.add_argument("--wrong", default="data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl")
    parser.add_argument("--blank", default="data/processed/lingoqa_clean_v2_train_blank_frame.jsonl")
    parser.add_argument("--normal_predictions", required=True)
    parser.add_argument("--text_predictions", required=True)
    parser.add_argument("--wrong_predictions", required=True)
    parser.add_argument("--blank_predictions", required=True)
    parser.add_argument("--output", default="data/processed/lingoqa_preference_v7_grounded.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_preference_v7_grounded_summary.json")
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--normal_weight", type=float, default=MODE_WEIGHTS["normal"])
    parser.add_argument("--text_only_weight", type=float, default=0.20)
    parser.add_argument("--blank_image_weight", type=float, default=0.25)
    parser.add_argument("--wrong_image_weight", type=float, default=0.35)
    parser.add_argument("--normal_sft_anchor_weight", type=float, default=1.5)
    parser.add_argument("--normal_bad_f1_threshold", type=float, default=0.15)
    parser.add_argument("--min_normal_prediction_f1_for_control", type=float, default=0.45)
    parser.add_argument("--max_control_prediction_f1", type=float, default=0.35)
    parser.add_argument("--min_visual_contrast_gap", type=float, default=0.20)
    parser.add_argument("--control_modes", default="wrong_image,blank_image")
    parser.add_argument("--max_text_only_pairs", type=int, default=0)
    parser.add_argument("--max_blank_image_pairs", type=int, default=80)
    parser.add_argument("--max_wrong_image_pairs", type=int, default=80)
    parser.add_argument("--max_control_pairs_per_capability", type=int, default=60)
    parser.add_argument(
        "--capability_sft_anchor_weights",
        default="spatial_localization=2.25,counting=1.75,object_recognition=1.5,reasoning_world_knowledge=1.75",
        help="Comma-separated capability=weight overrides for normal SFT anchors.",
    )
    parser.add_argument(
        "--capability_control_weight_multipliers",
        default="spatial_localization=0.65,counting=0.75,object_recognition=0.85",
        help="Comma-separated capability=multiplier overrides for control DPO weights.",
    )
    args = parser.parse_args()

    normal_rows = read_jsonl(ROOT / args.normal)
    wrong_rows = read_jsonl(ROOT / args.wrong)
    blank_rows = read_jsonl(ROOT / args.blank)
    normal_by_id = index_by_id(normal_rows)
    wrong_by_id = index_by_id(wrong_rows)
    blank_by_id = index_by_id(blank_rows)

    normal_predictions = index_by_id(read_jsonl(ROOT / args.normal_predictions))
    text_predictions = index_by_id(read_jsonl(ROOT / args.text_predictions))
    wrong_predictions = index_by_id(read_jsonl(ROOT / args.wrong_predictions))
    blank_predictions = index_by_id(read_jsonl(ROOT / args.blank_predictions))

    pairs: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    f1_sums: dict[str, float] = defaultdict(float)
    by_capability_added: Counter[str] = Counter()

    add_normal_pairs(pairs, normal_rows, normal_predictions, args, counters)
    enabled_modes = {item.strip() for item in args.control_modes.split(",") if item.strip()}
    if "text_only" in enabled_modes:
        add_control_pairs(pairs, "text_only", normal_by_id, text_predictions, normal_by_id, normal_predictions, args, counters, f1_sums, by_capability_added)
    if "wrong_image" in enabled_modes:
        add_control_pairs(pairs, "wrong_image", wrong_by_id, wrong_predictions, normal_by_id, normal_predictions, args, counters, f1_sums, by_capability_added)
    if "blank_image" in enabled_modes:
        add_control_pairs(pairs, "blank_image", blank_by_id, blank_predictions, normal_by_id, normal_predictions, args, counters, f1_sums, by_capability_added)

    random.Random(args.seed).shuffle(pairs)
    write_jsonl(ROOT / args.output, pairs)

    summary = {
        "recipe": "v7_grounded_visual_contrast",
        "output": args.output,
        "normal_rows": len(normal_rows),
        "preference_pairs": len(pairs),
        "control_modes": sorted(enabled_modes),
        "thresholds": {
            "min_normal_prediction_f1_for_control": args.min_normal_prediction_f1_for_control,
            "max_control_prediction_f1": args.max_control_prediction_f1,
            "min_visual_contrast_gap": args.min_visual_contrast_gap,
            "normal_bad_f1_threshold": args.normal_bad_f1_threshold,
        },
        "weights": {
            "normal_weight": args.normal_weight,
            "text_only_weight": args.text_only_weight,
            "wrong_image_weight": args.wrong_image_weight,
            "blank_image_weight": args.blank_image_weight,
            "normal_sft_anchor_weight": args.normal_sft_anchor_weight,
            "capability_sft_anchor_weights": parse_weight_map(args.capability_sft_anchor_weights),
            "capability_control_weight_multipliers": parse_weight_map(args.capability_control_weight_multipliers),
        },
        "by_pair_type": dict(Counter(row["pair_type"] for row in pairs)),
        "by_train_input_mode": dict(Counter(row["train_input_mode"] for row in pairs)),
        "by_capability": dict(Counter(row["meta"]["capability"] for row in pairs)),
        "control_pairs_by_capability": dict(by_capability_added),
        "weight_sum_by_mode": {
            mode: round(sum(float(row["weight"]) for row in pairs if row["train_input_mode"] == mode), 4)
            for mode in sorted({row["train_input_mode"] for row in pairs})
        },
        "sft_anchor_weight_sum_by_mode": {
            mode: round(sum(float(row["sft_anchor_weight"]) for row in pairs if row["train_input_mode"] == mode), 4)
            for mode in sorted({row["train_input_mode"] for row in pairs})
        },
        "counters": dict(counters),
        "avg_prediction_f1_by_control_mode": {
            mode: round(f1_sums[mode] / counters.get(f"{mode}_seen", 1), 4)
            for mode in ("text_only", "wrong_image", "blank_image")
            if counters.get(f"{mode}_seen", 0)
        },
        "notes": [
            "Control pairs require strong normal predictions and weak ablated predictions.",
            "Spatial localization receives higher normal anchors and lower control weights to reduce normal regression.",
            "Text-only controls are disabled by default because v5/v6 showed answer suppression without stable test gains.",
        ],
    }
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
