"""Build failure-mined preference pairs from SFT-v2 train predictions.

This replaces the v4 "refuse every control" recipe.  Control preference pairs
are created only when the current SFT-v2 adapter actually produced a
non-refusal answer on text-only / wrong-image / blank-image controls.  The
rejected response is that real model prediction, not the gold answer.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.output_parser import parse_model_output
from src.eval.refusal_detection import is_refusal_prediction
from src.eval.run_all_eval import token_f1


MODE_WEIGHTS = {
    "normal": 1.0,
    "text_only": 0.30,
    "blank_image": 0.40,
    "wrong_image": 0.60,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_obj(row: dict[str, Any]) -> dict[str, str]:
    answer = row.get("answer", {})
    if isinstance(answer, dict):
        text = str(answer.get("answer") or "").strip()
        reason = str(answer.get("reason") or text).strip()
        return {"task": str(answer.get("task") or "external_vqa"), "answer": text, "reason": reason}
    text = str(answer or "").strip()
    return {"task": "external_vqa", "answer": text, "reason": text}


def prediction_obj(prediction: Any) -> dict[str, str]:
    parsed = parse_model_output(prediction)
    data = parsed["data"] if parsed.get("ok") else {}
    if isinstance(data, dict) and (data.get("answer") or data.get("reason")):
        answer = str(data.get("answer") or data.get("reason") or "").strip()
        reason = str(data.get("reason") or answer).strip()
        return {"task": str(data.get("task") or "external_vqa"), "answer": answer, "reason": reason}
    text = str(prediction or "").strip()
    return {"task": "external_vqa", "answer": text, "reason": text}


def answer_text(obj: dict[str, Any]) -> str:
    return str(obj.get("answer") or obj.get("reason") or "")


def capability(row: dict[str, Any]) -> str:
    return str(
        row.get("meta", {}).get("external", {}).get("capability")
        or row.get("answer", {}).get("subcategory")
        or row.get("perception", {}).get("risk_hint")
        or "unknown"
    )


def prompt_sample(row: dict[str, Any], mode: str) -> dict[str, Any]:
    sample = copy.deepcopy(row)
    sample.pop("answer", None)
    sample.setdefault("meta", {})
    sample["meta"]["train_input_mode"] = mode
    return sample


def refusal_for(mode: str) -> dict[str, str]:
    if mode == "text_only":
        reason = "No image frames are available, so this visual driving question cannot be answered reliably."
    elif mode == "blank_image":
        reason = "The image frames are blank placeholders and contain no driving-scene evidence."
    elif mode == "wrong_image":
        reason = "The provided frames do not match the original scene for this question, so a scene-specific answer would be unreliable."
    else:
        reason = "The normal image input is answerable; refusing would discard usable visual evidence."
    return {
        "task": "external_vqa",
        "answer": "I cannot determine this from the provided visual input.",
        "reason": reason,
    }


def make_pair(
    row: dict[str, Any],
    chosen: dict[str, str],
    rejected: dict[str, str],
    pair_type: str,
    mode: str,
    weight: float,
    sft_anchor_weight: float,
    meta_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cap = capability(row)
    meta = {
        "source": "lingoqa_preference_v5_from_predictions",
        "capability": cap,
        "clean_split": row.get("meta", {}).get("clean_split", "train"),
        "weight": round(weight, 4),
        "sft_anchor_weight": round(sft_anchor_weight, 4),
    }
    if meta_extra:
        meta.update(meta_extra)
    return {
        "id": f"{row.get('id')}__{pair_type}__{mode}",
        "source_id": row.get("id"),
        "pair_type": pair_type,
        "train_input_mode": mode,
        "weight": round(weight, 4),
        "sft_anchor_weight": round(sft_anchor_weight, 4),
        "prompt_sample": prompt_sample(row, mode),
        "chosen": chosen,
        "rejected": rejected,
        "meta": meta,
    }


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


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

        rejected = refusal_for("normal")
        pair_type = "normal_answer_over_refusal"
        pred_row = normal_predictions.get(str(row.get("id"))) if normal_predictions else None
        if pred_row is not None:
            pred = prediction_obj(pred_row.get("prediction"))
            pred_f1 = token_f1(answer_text(pred), answer_text(gold))
            if is_refusal_prediction(pred_row.get("prediction")) or pred_f1 <= args.normal_bad_f1_threshold:
                rejected = pred
                pair_type = "normal_answer_over_bad_prediction"

        pairs.append(
            make_pair(
                row=row,
                chosen=gold,
                rejected=rejected,
                pair_type=pair_type,
                mode="normal",
                weight=args.normal_weight,
                sft_anchor_weight=args.normal_sft_anchor_weight,
                meta_extra={"rejected_source": "model_prediction" if pair_type.endswith("bad_prediction") else "template_refusal"},
            )
        )


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
) -> None:
    max_pairs = int(getattr(args, f"max_{mode}_pairs"))
    added = 0
    for sample_id, pred_row in predictions.items():
        if max_pairs > 0 and added >= max_pairs:
            counters[f"{mode}_skipped_max_pairs"] += 1
            continue
        source_row = source_by_id.get(sample_id)
        gold_row = gold_by_id.get(sample_id)
        if source_row is None or gold_row is None:
            counters[f"{mode}_skipped_missing_source"] += 1
            continue
        gold = answer_obj(gold_row)
        normal_pred_f1 = None
        if args.min_normal_prediction_f1_for_control > 0.0:
            normal_pred_row = normal_predictions.get(sample_id) if normal_predictions else None
            if normal_pred_row is None:
                counters[f"{mode}_skipped_missing_normal_prediction"] += 1
                continue
            normal_pred = prediction_obj(normal_pred_row.get("prediction"))
            normal_pred_f1 = token_f1(answer_text(normal_pred), answer_text(gold))
            if normal_pred_f1 < args.min_normal_prediction_f1_for_control:
                counters[f"{mode}_skipped_low_normal_prediction_f1"] += 1
                continue

        rejected = prediction_obj(pred_row.get("prediction"))
        pred_text = answer_text(rejected)
        pred_f1 = token_f1(pred_text, answer_text(gold))
        f1_sums[mode] += pred_f1
        counters[f"{mode}_seen"] += 1
        if is_refusal_prediction(pred_row.get("prediction")):
            counters[f"{mode}_skipped_already_refusal"] += 1
            continue
        if not pred_text.strip():
            counters[f"{mode}_skipped_empty_prediction"] += 1
            continue
        if pred_f1 < args.min_control_f1 and not args.include_all_non_refusal_controls:
            counters[f"{mode}_skipped_low_overlap"] += 1
            continue

        pairs.append(
            make_pair(
                row=source_row,
                chosen=refusal_for(mode),
                rejected=rejected,
                pair_type="control_refusal_over_model_hallucination",
                mode=mode,
                weight=float(getattr(args, f"{mode}_weight")),
                sft_anchor_weight=0.0,
                meta_extra={
                    "rejected_source": "sft_v2_prediction",
                    "prediction_f1_vs_gold": round(pred_f1, 4),
                    "normal_prediction_f1_vs_gold": round(normal_pred_f1, 4) if normal_pred_f1 is not None else None,
                    "prediction_was_refusal": False,
                },
            )
        )
        counters[f"{mode}_added"] += 1
        added += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v5 preference pairs from SFT-v2 train predictions.")
    parser.add_argument("--normal", default="data/processed/lingoqa_clean_v2_train.jsonl")
    parser.add_argument("--wrong", default="data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl")
    parser.add_argument("--blank", default="data/processed/lingoqa_clean_v2_train_blank_frame.jsonl")
    parser.add_argument("--normal_predictions", default="")
    parser.add_argument("--text_predictions", required=True)
    parser.add_argument("--wrong_predictions", required=True)
    parser.add_argument("--blank_predictions", required=True)
    parser.add_argument("--output", default="data/processed/lingoqa_preference_v5_from_sft_v2_train_predictions.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_preference_v5_from_sft_v2_train_predictions_summary.json")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--normal_weight", type=float, default=MODE_WEIGHTS["normal"])
    parser.add_argument("--text_only_weight", type=float, default=MODE_WEIGHTS["text_only"])
    parser.add_argument("--blank_image_weight", type=float, default=MODE_WEIGHTS["blank_image"])
    parser.add_argument("--wrong_image_weight", type=float, default=MODE_WEIGHTS["wrong_image"])
    parser.add_argument("--normal_sft_anchor_weight", type=float, default=1.0)
    parser.add_argument("--normal_bad_f1_threshold", type=float, default=0.15)
    parser.add_argument("--min_control_f1", type=float, default=0.05)
    parser.add_argument(
        "--min_normal_prediction_f1_for_control",
        type=float,
        default=0.0,
        help="Only mine control pairs for samples whose normal prediction reaches this F1.",
    )
    parser.add_argument("--include_all_non_refusal_controls", action="store_true", default=True)
    parser.add_argument(
        "--filter_low_overlap_controls",
        action="store_false",
        dest="include_all_non_refusal_controls",
        help="Respect --min_control_f1 instead of keeping every non-refusal control prediction.",
    )
    parser.add_argument(
        "--control_modes",
        default="text_only,wrong_image,blank_image",
        help="Comma-separated control modes to mine: text_only,wrong_image,blank_image.",
    )
    parser.add_argument("--max_text_only_pairs", type=int, default=0)
    parser.add_argument("--max_blank_image_pairs", type=int, default=0)
    parser.add_argument("--max_wrong_image_pairs", type=int, default=0)
    args = parser.parse_args()

    normal_rows = read_jsonl(ROOT / args.normal)
    wrong_rows = read_jsonl(ROOT / args.wrong)
    blank_rows = read_jsonl(ROOT / args.blank)
    normal_by_id = index_by_id(normal_rows)
    wrong_by_id = index_by_id(wrong_rows)
    blank_by_id = index_by_id(blank_rows)

    normal_predictions = index_by_id(read_jsonl(ROOT / args.normal_predictions)) if args.normal_predictions else None
    text_predictions = index_by_id(read_jsonl(ROOT / args.text_predictions))
    wrong_predictions = index_by_id(read_jsonl(ROOT / args.wrong_predictions))
    blank_predictions = index_by_id(read_jsonl(ROOT / args.blank_predictions))

    pairs: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    f1_sums: dict[str, float] = defaultdict(float)

    add_normal_pairs(pairs, normal_rows, normal_predictions, args, counters)
    enabled_modes = {item.strip() for item in args.control_modes.split(",") if item.strip()}
    if "text_only" in enabled_modes:
        add_control_pairs(pairs, "text_only", normal_by_id, text_predictions, normal_by_id, normal_predictions, args, counters, f1_sums)
    if "wrong_image" in enabled_modes:
        add_control_pairs(pairs, "wrong_image", wrong_by_id, wrong_predictions, normal_by_id, normal_predictions, args, counters, f1_sums)
    if "blank_image" in enabled_modes:
        add_control_pairs(pairs, "blank_image", blank_by_id, blank_predictions, normal_by_id, normal_predictions, args, counters, f1_sums)

    random.Random(args.seed).shuffle(pairs)
    write_jsonl(ROOT / args.output, pairs)

    summary = {
        "normal": args.normal,
        "wrong": args.wrong,
        "blank": args.blank,
        "normal_predictions": args.normal_predictions,
        "text_predictions": args.text_predictions,
        "wrong_predictions": args.wrong_predictions,
        "blank_predictions": args.blank_predictions,
        "control_modes": sorted(enabled_modes),
        "min_control_f1": args.min_control_f1,
        "include_all_non_refusal_controls": args.include_all_non_refusal_controls,
        "min_normal_prediction_f1_for_control": args.min_normal_prediction_f1_for_control,
        "output": args.output,
        "normal_rows": len(normal_rows),
        "preference_pairs": len(pairs),
        "by_pair_type": dict(Counter(row["pair_type"] for row in pairs)),
        "by_train_input_mode": dict(Counter(row["train_input_mode"] for row in pairs)),
        "by_capability": dict(Counter(row["meta"]["capability"] for row in pairs)),
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
            "Control pairs are mined only from SFT-v2 non-refusal predictions.",
            "Control pairs have sft_anchor_weight=0 to avoid SFT-training refusal templates.",
            "Normal pairs keep sft_anchor_weight>0 to preserve answerability.",
        ],
    }
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
