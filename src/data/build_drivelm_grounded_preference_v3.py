"""Build conservative DriveLM grounded preference pairs.

Unlike the earlier refusal-heavy recipe, this builder only keeps control pairs
when the normal-image answer is strong and the ablated-input answer is clearly
weak. This makes the preference signal image-contrastive instead of a blanket
refusal policy.
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

from src.agent.output_parser import parse_model_output
from src.eval.refusal_detection import is_refusal_prediction
from src.eval.run_all_eval import token_f1


CONTROL_MODES = ("wrong_image", "blank_image", "text_only")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in rows}


def parse_answer_obj(value: Any) -> dict[str, str]:
    parsed = parse_model_output(value)
    if parsed.get("ok") and isinstance(parsed.get("data"), dict):
        data = parsed["data"]
        text = str(data.get("answer") or data.get("reason") or "").strip()
        reason = str(data.get("reason") or text).strip()
        return {"task": str(data.get("task") or "external_vqa"), "answer": text, "reason": reason}
    text = str(value or "").strip()
    return {"task": "external_vqa", "answer": text, "reason": text}


def gold_answer(row: dict[str, Any]) -> dict[str, str]:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else row.get("gold", {})
    if not isinstance(answer, dict):
        text = str(answer or "").strip()
        return {"task": "external_vqa", "answer": text, "reason": text}
    text = str(answer.get("answer") or answer.get("reason") or "").strip()
    reason = str(answer.get("reason") or text).strip()
    return {"task": str(answer.get("task") or "external_vqa"), "answer": text, "reason": reason}


def pred_answer(row: dict[str, Any] | None) -> dict[str, str]:
    if not row:
        return {"task": "external_vqa", "answer": "", "reason": ""}
    return parse_answer_obj(row.get("prediction"))


def answer_text(answer: dict[str, Any]) -> str:
    return str(answer.get("answer") or answer.get("reason") or "")


def uncertainty_for(mode: str) -> dict[str, str]:
    if mode == "blank_image":
        reason = "The provided frames contain no usable driving-scene evidence, so a scene-specific answer would be unreliable."
    elif mode == "text_only":
        reason = "No visual frames are available, so this visual driving question cannot be answered reliably."
    elif mode == "wrong_image":
        reason = "The provided frames do not give reliable visual evidence for this scene-specific question."
    else:
        reason = "The normal visual input contains usable evidence, so refusing would discard relevant information."
    return {
        "task": "external_vqa",
        "answer": "I cannot determine this from the provided visual input.",
        "reason": reason,
    }


def deep_copy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def prompt_sample(row: dict[str, Any], mode: str) -> dict[str, Any]:
    sample = deep_copy_json(row)
    sample.pop("answer", None)
    sample.setdefault("meta", {})
    sample["meta"]["train_input_mode"] = mode
    sample["meta"]["recommended_perception_mode"] = "none"
    return sample


def capability(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    return str(meta.get("capability") or external.get("capability") or answer.get("subcategory") or "unknown")


def category(row: dict[str, Any]) -> str:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    external = row.get("meta", {}).get("external", {}) if isinstance(row.get("meta"), dict) else {}
    return str(answer.get("category") or external.get("category") or "unknown")


def make_pair(
    *,
    pair_id: str,
    source_id: str,
    source_row: dict[str, Any],
    chosen: dict[str, str],
    rejected: dict[str, str],
    pair_type: str,
    mode: str,
    weight: float,
    sft_anchor_weight: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    merged_meta = {
        "source": "drivelm_grounded_preference_v3",
        "capability": capability(source_row),
        "category": category(source_row),
        "weight": round(weight, 4),
        "sft_anchor_weight": round(sft_anchor_weight, 4),
        "recommended_perception_mode": "none",
    }
    merged_meta.update(meta)
    return {
        "id": pair_id,
        "source_id": source_id,
        "pair_type": pair_type,
        "train_input_mode": mode,
        "weight": round(weight, 4),
        "sft_anchor_weight": round(sft_anchor_weight, 4),
        "prompt_sample": prompt_sample(source_row, mode),
        "chosen": chosen,
        "rejected": rejected,
        "meta": merged_meta,
    }


def case_score(case: dict[str, Any], mode: str) -> float:
    return float(case.get(f"{mode}_f1", 0.0) or 0.0)


def parse_weight_map(spec: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"invalid weight map item: {item}")
        key, value = item.split("=", 1)
        result[key.strip()] = float(value)
    return result


def cap_anchor_weight(cap: str, args: argparse.Namespace) -> float:
    return parse_weight_map(args.capability_sft_anchor_weights).get(cap, args.normal_sft_anchor_weight)


def cap_control_multiplier(cap: str, args: argparse.Namespace) -> float:
    return parse_weight_map(args.capability_control_weight_multipliers).get(cap, 1.0)


def add_normal_anchor_pairs(
    pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    rng = random.Random(args.seed)
    anchor_rows = list(rows)
    rng.shuffle(anchor_rows)
    if args.max_normal_anchor_pairs > 0:
        anchor_rows = anchor_rows[: args.max_normal_anchor_pairs]
    for row in anchor_rows:
        gold = gold_answer(row)
        if not answer_text(gold):
            counters["normal_anchor_skipped_empty_gold"] += 1
            continue
        source_id = str(row.get("id"))
        cap = capability(row)
        pairs.append(
            make_pair(
                pair_id=f"{source_id}__normal_gold_over_uncertainty",
                source_id=source_id,
                source_row=row,
                chosen=gold,
                rejected=uncertainty_for("normal"),
                pair_type="normal_gold_over_uncertainty",
                mode="normal",
                weight=args.normal_weight,
                sft_anchor_weight=cap_anchor_weight(cap, args),
                meta={"rejected_source": "template_uncertainty"},
            )
        )
        counters["normal_anchor_added"] += 1


def add_control_pairs(
    pairs: list[dict[str, Any]],
    mode: str,
    cases: list[dict[str, Any]],
    normal_rows: dict[str, dict[str, Any]],
    control_rows: dict[str, dict[str, Any]],
    control_preds: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    counters: Counter[str],
    by_capability_added: Counter[str],
) -> None:
    for case in sorted(cases, key=lambda item: float(item.get("visual_dependency_gap", 0.0) or 0.0)):
        sample_id = str(case.get("id") or "")
        normal_row = normal_rows.get(sample_id)
        control_row = control_rows.get(sample_id) if mode != "text_only" else normal_row
        pred_row = control_preds.get(sample_id)
        if normal_row is None or control_row is None or pred_row is None:
            counters[f"{mode}_skipped_missing_source"] += 1
            continue
        cap = capability(normal_row)
        if args.max_control_pairs_per_capability > 0 and by_capability_added[cap] >= args.max_control_pairs_per_capability:
            counters[f"{mode}_skipped_capability_quota"] += 1
            continue
        normal_f1 = case_score(case, "normal")
        control_f1 = case_score(case, mode)
        if normal_f1 < args.min_normal_f1:
            counters[f"{mode}_skipped_low_normal_f1"] += 1
            continue
        if control_f1 > args.max_control_f1:
            counters[f"{mode}_skipped_high_control_f1"] += 1
            continue
        if normal_f1 - control_f1 < args.min_visual_contrast_gap:
            counters[f"{mode}_skipped_low_visual_contrast_gap"] += 1
            continue
        if is_refusal_prediction(pred_row.get("prediction")):
            counters[f"{mode}_skipped_already_uncertain"] += 1
            continue
        rejected = pred_answer(pred_row)
        if not answer_text(rejected).strip():
            counters[f"{mode}_skipped_empty_prediction"] += 1
            continue

        source_id = str(normal_row.get("id"))
        weight = float(getattr(args, f"{mode}_weight")) * cap_control_multiplier(cap, args)
        pairs.append(
            make_pair(
                pair_id=f"{source_id}__{mode}_uncertainty_over_ablated_prediction",
                source_id=source_id,
                source_row=control_row,
                chosen=uncertainty_for(mode),
                rejected=rejected,
                pair_type="control_uncertainty_over_ablated_prediction",
                mode=mode,
                weight=weight,
                sft_anchor_weight=0.0,
                meta={
                    "control_mode": mode,
                    "normal_f1": round(normal_f1, 4),
                    "control_f1": round(control_f1, 4),
                    "visual_contrast_gap": round(normal_f1 - control_f1, 4),
                    "rejected_source": f"{mode}_prediction",
                },
            )
        )
        counters[f"{mode}_added"] += 1
        by_capability_added[cap] += 1


def build_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normal_rows = read_jsonl(Path(args.normal_data))
    wrong_rows = read_jsonl(Path(args.wrong_data)) if args.wrong_data else []
    blank_rows = read_jsonl(Path(args.blank_data)) if args.blank_data else []
    cases = read_jsonl(Path(args.cases))
    predictions = {
        "text_only": index_by_id(read_jsonl(Path(args.text_only_predictions))),
        "wrong_image": index_by_id(read_jsonl(Path(args.wrong_image_predictions))),
        "blank_image": index_by_id(read_jsonl(Path(args.blank_image_predictions))),
    }
    normal_by_id = index_by_id(normal_rows)
    wrong_by_id = index_by_id(wrong_rows)
    blank_by_id = index_by_id(blank_rows)
    source_by_mode = {
        "text_only": normal_by_id,
        "wrong_image": wrong_by_id,
        "blank_image": blank_by_id,
    }
    counters: Counter[str] = Counter()
    by_capability_added: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []

    add_normal_anchor_pairs(pairs, normal_rows, args, counters)
    enabled_modes = {item.strip() for item in args.control_modes.split(",") if item.strip()}
    for mode in CONTROL_MODES:
        if mode not in enabled_modes:
            continue
        add_control_pairs(
            pairs,
            mode,
            cases,
            normal_by_id,
            source_by_mode[mode],
            predictions[mode],
            args,
            counters,
            by_capability_added,
        )

    random.Random(args.seed).shuffle(pairs)
    by_pair_type = Counter(str(pair.get("pair_type")) for pair in pairs)
    by_mode = Counter(str(pair.get("train_input_mode")) for pair in pairs)
    by_capability = Counter(str(pair.get("meta", {}).get("capability")) for pair in pairs)
    summary = {
        "recipe": "drivelm_grounded_preference_v3",
        "pairs": len(pairs),
        "normal_rows": len(normal_rows),
        "case_rows": len(cases),
        "control_modes": sorted(enabled_modes),
        "thresholds": {
            "min_normal_f1": args.min_normal_f1,
            "max_control_f1": args.max_control_f1,
            "min_visual_contrast_gap": args.min_visual_contrast_gap,
        },
        "by_pair_type": dict(by_pair_type.most_common()),
        "by_train_input_mode": dict(by_mode.most_common()),
        "by_capability": dict(by_capability.most_common()),
        "control_pairs_by_capability": dict(by_capability_added.most_common()),
        "weight_sum_by_mode": {
            mode: round(sum(float(pair["weight"]) for pair in pairs if pair["train_input_mode"] == mode), 4)
            for mode in sorted(by_mode)
        },
        "sft_anchor_weight_sum_by_mode": {
            mode: round(
                sum(float(pair["sft_anchor_weight"]) for pair in pairs if pair["train_input_mode"] == mode), 4
            )
            for mode in sorted(by_mode)
        },
        "counters": dict(counters.most_common()),
        "inputs": {
            "normal_data": args.normal_data,
            "wrong_data": args.wrong_data,
            "blank_data": args.blank_data,
            "cases": args.cases,
            "text_only_predictions": args.text_only_predictions,
            "wrong_image_predictions": args.wrong_image_predictions,
            "blank_image_predictions": args.blank_image_predictions,
        },
    }
    return pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conservative DriveLM grounded preference pairs.")
    parser.add_argument("--normal_data", default="data/processed/drivelm_train_scene.jsonl")
    parser.add_argument("--wrong_data", default="data/processed/drivelm_train_scene_wrong_image.jsonl")
    parser.add_argument("--blank_data", default="data/processed/drivelm_train_scene_blank_image.jsonl")
    parser.add_argument("--cases", default="outputs/cases/drivelm_train_scene_sft_1200_visual_control_cases.jsonl")
    parser.add_argument("--text_only_predictions", default="outputs/eval_results/drivelm_train_scene_sft_1200_text_only_predictions.jsonl")
    parser.add_argument("--wrong_image_predictions", default="outputs/eval_results/drivelm_train_scene_sft_1200_wrong_predictions.jsonl")
    parser.add_argument("--blank_image_predictions", default="outputs/eval_results/drivelm_train_scene_sft_1200_blank_predictions.jsonl")
    parser.add_argument("--output", default="data/processed/drivelm_grounded_pref_v3_train.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/drivelm_grounded_pref_v3_train_summary.json")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--control_modes", default="wrong_image,blank_image")
    parser.add_argument("--max_normal_anchor_pairs", type=int, default=600)
    parser.add_argument("--max_control_pairs_per_capability", type=int, default=120)
    parser.add_argument("--min_normal_f1", type=float, default=0.40)
    parser.add_argument("--max_control_f1", type=float, default=0.30)
    parser.add_argument("--min_visual_contrast_gap", type=float, default=0.20)
    parser.add_argument("--normal_weight", type=float, default=1.0)
    parser.add_argument("--text_only_weight", type=float, default=0.15)
    parser.add_argument("--wrong_image_weight", type=float, default=0.30)
    parser.add_argument("--blank_image_weight", type=float, default=0.25)
    parser.add_argument("--normal_sft_anchor_weight", type=float, default=1.5)
    parser.add_argument(
        "--capability_sft_anchor_weights",
        default="spatial_localization=2.0,reasoning_world_knowledge=1.8,object_recognition=1.6,counting=1.6",
    )
    parser.add_argument(
        "--capability_control_weight_multipliers",
        default="spatial_localization=0.65,reasoning_world_knowledge=0.9,object_recognition=0.85,counting=0.8",
    )
    args = parser.parse_args()

    pairs, summary = build_pairs(args)
    write_jsonl(Path(args.output), pairs)
    write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output} and {args.summary}")


if __name__ == "__main__":
    main()
