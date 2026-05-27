"""Build DriveLM control-abstention preference pairs.

This recipe is intentionally less conservative than grounded_preference_v3.
It targets the failure mode where text-only, blank-image, or hard wrong-image
controls can still answer the original question from language priors. Normal
inputs prefer the gold answer; control inputs prefer an uncertainty response
over the original gold answer.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


CONTROL_MODES = ("text_only", "wrong_image", "blank_image")


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


def deep_copy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def gold_answer(row: dict[str, Any]) -> dict[str, str]:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else row.get("gold", {})
    if not isinstance(answer, dict):
        text = str(answer or "").strip()
        return {"task": "external_vqa", "answer": text, "reason": text}
    text = str(answer.get("answer") or answer.get("reason") or "").strip()
    reason = str(answer.get("reason") or text).strip()
    return {"task": str(answer.get("task") or "external_vqa"), "answer": text, "reason": reason}


def answer_text(answer: dict[str, Any]) -> str:
    return str(answer.get("answer") or answer.get("reason") or "")


def uncertainty_for(mode: str) -> dict[str, str]:
    if mode == "text_only":
        reason = "No visual frames are available, so this visual driving question cannot be answered reliably."
    elif mode == "blank_image":
        reason = "The provided frames contain no usable driving-scene evidence, so a scene-specific answer would be unreliable."
    elif mode == "wrong_image":
        reason = "The provided frames do not provide reliable evidence for the queried driving scene."
    else:
        reason = "The normal visual input contains usable evidence for this scene-specific question."
    return {
        "task": "external_vqa",
        "answer": "I cannot determine this from the provided visual input.",
        "reason": reason,
    }


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


def mode_weight(mode: str, args: argparse.Namespace) -> float:
    return {
        "normal": args.normal_weight,
        "text_only": args.text_only_weight,
        "wrong_image": args.wrong_image_weight,
        "blank_image": args.blank_image_weight,
    }[mode]


def cap_multiplier(cap: str, args: argparse.Namespace) -> float:
    return parse_weight_map(args.capability_weight_multipliers).get(cap, 1.0)


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
        "source": "drivelm_control_abstention_pref_v4",
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


def add_normal_pairs(
    pairs: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    for row in rows:
        gold = gold_answer(row)
        if not answer_text(gold):
            counters["normal_skipped_empty_gold"] += 1
            continue
        source_id = str(row.get("id"))
        cap = capability(row)
        pairs.append(
            make_pair(
                pair_id=f"{source_id}__normal_gold_over_uncertainty_v4",
                source_id=source_id,
                source_row=row,
                chosen=gold,
                rejected=uncertainty_for("normal"),
                pair_type="normal_gold_over_uncertainty",
                mode="normal",
                weight=mode_weight("normal", args) * cap_multiplier(cap, args),
                sft_anchor_weight=args.normal_sft_anchor_weight,
                meta={"rejected_source": "template_uncertainty"},
            )
        )
        counters["normal_added"] += 1


def add_control_pairs(
    pairs: list[dict[str, Any]],
    *,
    mode: str,
    normal_rows: list[dict[str, Any]],
    control_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    per_cap: Counter[str] = Counter()
    for normal_row in normal_rows:
        source_id = str(normal_row.get("id"))
        control_row = normal_row if mode == "text_only" else control_by_id.get(source_id)
        if control_row is None:
            counters[f"{mode}_skipped_missing_control"] += 1
            continue
        gold = gold_answer(normal_row)
        if not answer_text(gold):
            counters[f"{mode}_skipped_empty_gold"] += 1
            continue
        cap = capability(normal_row)
        if args.max_control_pairs_per_capability > 0 and per_cap[cap] >= args.max_control_pairs_per_capability:
            counters[f"{mode}_skipped_capability_quota"] += 1
            continue
        weight = mode_weight(mode, args) * cap_multiplier(cap, args)
        pairs.append(
            make_pair(
                pair_id=f"{source_id}__{mode}_uncertainty_over_gold_v4",
                source_id=source_id,
                source_row=control_row,
                chosen=uncertainty_for(mode),
                rejected=gold,
                pair_type="control_uncertainty_over_gold",
                mode=mode,
                weight=weight,
                sft_anchor_weight=args.control_sft_anchor_weight,
                meta={"control_mode": mode, "rejected_source": "original_gold"},
            )
        )
        counters[f"{mode}_added"] += 1
        per_cap[cap] += 1


def limited_shuffle(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    out = list(rows)
    random.Random(seed).shuffle(out)
    if limit > 0:
        out = out[:limit]
    return out


def build_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normal_rows_all = read_jsonl(Path(args.normal_data))
    wrong_rows = read_jsonl(Path(args.wrong_data)) if args.wrong_data else []
    blank_rows = read_jsonl(Path(args.blank_data)) if args.blank_data else []
    wrong_by_id = index_by_id(wrong_rows)
    blank_by_id = index_by_id(blank_rows)
    enabled_modes = {item.strip() for item in args.control_modes.split(",") if item.strip()}

    counters: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []
    normal_rows = limited_shuffle(normal_rows_all, args.max_normal_pairs, args.seed)
    control_rows = limited_shuffle(normal_rows_all, args.max_control_pairs_per_mode, args.seed + 17)

    add_normal_pairs(pairs, normal_rows, args, counters)
    if "text_only" in enabled_modes:
        add_control_pairs(pairs, mode="text_only", normal_rows=control_rows, control_by_id={}, args=args, counters=counters)
    if "wrong_image" in enabled_modes:
        add_control_pairs(
            pairs, mode="wrong_image", normal_rows=control_rows, control_by_id=wrong_by_id, args=args, counters=counters
        )
    if "blank_image" in enabled_modes:
        add_control_pairs(
            pairs, mode="blank_image", normal_rows=control_rows, control_by_id=blank_by_id, args=args, counters=counters
        )

    random.Random(args.seed + 101).shuffle(pairs)
    by_type = Counter(str(pair.get("pair_type")) for pair in pairs)
    by_mode = Counter(str(pair.get("train_input_mode")) for pair in pairs)
    by_capability = Counter(str(pair.get("meta", {}).get("capability")) for pair in pairs)
    summary = {
        "recipe": "drivelm_control_abstention_pref_v4",
        "pairs": len(pairs),
        "normal_rows": len(normal_rows_all),
        "control_modes": sorted(enabled_modes),
        "limits": {
            "max_normal_pairs": args.max_normal_pairs,
            "max_control_pairs_per_mode": args.max_control_pairs_per_mode,
            "max_control_pairs_per_capability": args.max_control_pairs_per_capability,
        },
        "by_pair_type": dict(by_type.most_common()),
        "by_train_input_mode": dict(by_mode.most_common()),
        "by_capability": dict(by_capability.most_common()),
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
        },
    }
    return pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM control-abstention preference pairs.")
    parser.add_argument("--normal_data", default="data/processed/drivelm_train_scene.jsonl")
    parser.add_argument("--wrong_data", default="data/processed/drivelm_train_scene_wrong_image_hard.jsonl")
    parser.add_argument("--blank_data", default="data/processed/drivelm_train_scene_blank_image.jsonl")
    parser.add_argument("--output", default="data/processed/drivelm_control_abstention_pref_v4_train.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/drivelm_control_abstention_pref_v4_train_summary.json")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--control_modes", default="text_only,wrong_image,blank_image")
    parser.add_argument("--max_normal_pairs", type=int, default=900)
    parser.add_argument("--max_control_pairs_per_mode", type=int, default=700)
    parser.add_argument("--max_control_pairs_per_capability", type=int, default=220)
    parser.add_argument("--normal_weight", type=float, default=1.0)
    parser.add_argument("--text_only_weight", type=float, default=0.45)
    parser.add_argument("--wrong_image_weight", type=float, default=0.45)
    parser.add_argument("--blank_image_weight", type=float, default=0.55)
    parser.add_argument("--normal_sft_anchor_weight", type=float, default=1.2)
    parser.add_argument("--control_sft_anchor_weight", type=float, default=0.8)
    parser.add_argument(
        "--capability_weight_multipliers",
        default="spatial_localization=1.15,reasoning_world_knowledge=1.0,object_recognition=1.0,counting=1.0",
    )
    args = parser.parse_args()

    pairs, summary = build_pairs(args)
    write_jsonl(Path(args.output), pairs)
    write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output} and {args.summary}")


if __name__ == "__main__":
    main()
