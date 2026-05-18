"""Build guarded preference pairs for post-SFT visual grounding.

Compared with the earlier control-heavy v3 set, this version keeps a normal
answer-vs-refusal pair for every training example. Controls are still dominant,
but they are weighted rather than blindly duplicated. This is meant to reduce
wrong/blank/text hallucination without turning the model into a blanket refuser.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

REFUSAL = {
    "task": "external_vqa",
    "answer": "I cannot determine this from the provided visual input.",
    "reason": "The available visual input does not provide reliable evidence for this answer, so I should not guess.",
}

MODE_WEIGHTS = {
    "normal": 0.7,
    "text_only": 1.0,
    "blank_image": 1.0,
    "wrong_image": 1.35,
}

CAPABILITY_CONTROL_BOOST = {
    "counting": 1.25,
    "spatial_localization": 1.15,
    "reasoning_world_knowledge": 1.10,
    "object_recognition": 0.90,
    "other": 1.00,
    "weather_road_condition": 1.00,
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
    text = str(answer).strip()
    return {"task": "external_vqa", "answer": text, "reason": text}


def capability(row: dict[str, Any]) -> str:
    return str(
        row.get("meta", {}).get("external", {}).get("capability")
        or row.get("answer", {}).get("subcategory")
        or row.get("perception", {}).get("risk_hint")
        or "unknown"
    )


def refusal_for(mode: str) -> dict[str, str]:
    if mode == "text_only":
        reason = "No image frames are available, so this visual driving question cannot be answered reliably."
    elif mode == "blank_image":
        reason = "The image frames are blank placeholders and contain no driving-scene evidence."
    elif mode == "wrong_image":
        reason = "The provided frames do not match the original scene for this question, so a scene-specific answer would be unreliable."
    else:
        reason = REFUSAL["reason"]
    return {"task": "external_vqa", "answer": REFUSAL["answer"], "reason": reason}


def prompt_sample(row: dict[str, Any], mode: str) -> dict[str, Any]:
    sample = copy.deepcopy(row)
    sample.pop("answer", None)
    sample.setdefault("meta", {})
    sample["meta"]["train_input_mode"] = mode
    return sample


def pair(
    row: dict[str, Any],
    chosen: dict[str, str],
    rejected: dict[str, str],
    pair_type: str,
    mode: str,
    suffix: str,
    weight: float,
) -> dict[str, Any]:
    cap = capability(row)
    return {
        "id": f"{row.get('id')}__{pair_type}__{suffix}",
        "source_id": row.get("id"),
        "pair_type": pair_type,
        "train_input_mode": mode,
        "weight": round(weight, 4),
        "prompt_sample": prompt_sample(row, mode),
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "source": "lingoqa_preference_v4_guarded",
            "capability": cap,
            "clean_split": row.get("meta", {}).get("clean_split", "train"),
            "weight": round(weight, 4),
        },
    }


def control_weight(mode: str, cap: str) -> float:
    if mode == "normal":
        return MODE_WEIGHTS["normal"]
    return MODE_WEIGHTS[mode] * CAPABILITY_CONTROL_BOOST.get(cap, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build guarded preference pairs for clean LingoQA train split.")
    parser.add_argument("--normal", default="data/processed/lingoqa_clean_v2_train.jsonl")
    parser.add_argument("--wrong", default="data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl")
    parser.add_argument("--blank", default="data/processed/lingoqa_clean_v2_train_blank_frame.jsonl")
    parser.add_argument("--output", default="data/processed/lingoqa_preference_v4_guarded_train.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_preference_v4_guarded_train_summary.json")
    parser.add_argument("--wrong_repeat", type=int, default=2)
    parser.add_argument("--blank_repeat", type=int, default=1)
    parser.add_argument("--text_repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    normal_rows = read_jsonl(ROOT / args.normal)
    wrong_by_id = {row["id"]: row for row in read_jsonl(ROOT / args.wrong)}
    blank_by_id = {row["id"]: row for row in read_jsonl(ROOT / args.blank)}

    pairs: list[dict[str, Any]] = []
    for row in normal_rows:
        cap = capability(row)
        gold = answer_obj(row)
        if not gold["answer"]:
            continue

        pairs.append(
            pair(
                row=row,
                chosen=gold,
                rejected=REFUSAL,
                pair_type="normal_answer_over_refusal",
                mode="normal",
                suffix="guard",
                weight=control_weight("normal", cap),
            )
        )

        control_specs = [
            ("text_only", row, args.text_repeat),
            ("blank_image", blank_by_id.get(str(row.get("id"))), args.blank_repeat),
            ("wrong_image", wrong_by_id.get(str(row.get("id"))), args.wrong_repeat),
        ]
        for mode, control_row, repeat in control_specs:
            if control_row is None:
                continue
            for rep in range(repeat):
                pairs.append(
                    pair(
                        row=control_row,
                        chosen=refusal_for(mode),
                        rejected=gold,
                        pair_type="control_refusal_over_hallucination",
                        mode=mode,
                        suffix=f"rep{rep:02d}",
                        weight=control_weight(mode, cap),
                    )
                )

    rng.shuffle(pairs)
    write_jsonl(ROOT / args.output, pairs)

    summary = {
        "normal": args.normal,
        "wrong": args.wrong,
        "blank": args.blank,
        "output": args.output,
        "normal_rows": len(normal_rows),
        "preference_pairs": len(pairs),
        "repeats": {"text_only": args.text_repeat, "blank_image": args.blank_repeat, "wrong_image": args.wrong_repeat},
        "mode_weights": MODE_WEIGHTS,
        "capability_control_boost": CAPABILITY_CONTROL_BOOST,
        "by_pair_type": dict(Counter(row["pair_type"] for row in pairs)),
        "by_train_input_mode": dict(Counter(row["train_input_mode"] for row in pairs)),
        "by_capability": dict(Counter(row["meta"]["capability"] for row in pairs)),
        "weight_sum_by_mode": {
            mode: round(sum(float(row["weight"]) for row in pairs if row["train_input_mode"] == mode), 4)
            for mode in sorted({row["train_input_mode"] for row in pairs})
        },
    }
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
