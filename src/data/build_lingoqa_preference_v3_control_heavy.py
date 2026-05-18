"""Build control-heavy preference pairs for clean LingoQA train split."""

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
        return {
            "task": str(answer.get("task") or "external_vqa"),
            "answer": str(answer.get("answer") or ""),
            "reason": str(answer.get("reason") or answer.get("answer") or ""),
        }
    return {"task": "external_vqa", "answer": str(answer), "reason": str(answer)}


def refusal_for(mode: str) -> dict[str, str]:
    if mode == "text_only":
        reason = "No image frames are available, so this visual driving question cannot be answered reliably."
    elif mode == "blank_image":
        reason = "The image frames are blank placeholders and contain no driving-scene evidence."
    elif mode == "wrong_image":
        reason = "The frames do not match the original scene for this question, so a scene-specific answer would be unreliable."
    else:
        reason = REFUSAL["reason"]
    return {"task": "external_vqa", "answer": REFUSAL["answer"], "reason": reason}


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


def pair(row: dict[str, Any], chosen: dict[str, str], rejected: dict[str, str], pair_type: str, mode: str, suffix: str) -> dict[str, Any]:
    return {
        "id": f"{row.get('id')}__{pair_type}__{suffix}",
        "source_id": row.get("id"),
        "pair_type": pair_type,
        "train_input_mode": mode,
        "prompt_sample": prompt_sample(row, mode),
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "source": "lingoqa_preference_v3_control_heavy",
            "capability": capability(row),
            "clean_split": row.get("meta", {}).get("clean_split", "train"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build control-heavy LingoQA preference pairs.")
    parser.add_argument("--normal", default="data/processed/lingoqa_clean_v2_train.jsonl")
    parser.add_argument("--wrong", default="data/processed/lingoqa_clean_v2_train_wrong_frame.jsonl")
    parser.add_argument("--blank", default="data/processed/lingoqa_clean_v2_train_blank_frame.jsonl")
    parser.add_argument("--output", default="data/processed/lingoqa_preference_v3_control_heavy_train.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_preference_v3_control_heavy_train_summary.json")
    parser.add_argument("--control_repeat", type=int, default=3)
    parser.add_argument("--normal_keep_ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    normal_rows = read_jsonl(ROOT / args.normal)
    wrong_by_id = {row["id"]: row for row in read_jsonl(ROOT / args.wrong)}
    blank_by_id = {row["id"]: row for row in read_jsonl(ROOT / args.blank)}

    pairs: list[dict[str, Any]] = []
    for row in normal_rows:
        rejected_answer = answer_obj(row)
        if rng.random() < args.normal_keep_ratio:
            pairs.append(pair(row, rejected_answer, REFUSAL, "normal_answer_over_refusal", "normal", "keep"))

        control_sources = [
            ("text_only", row),
            ("blank_image", blank_by_id.get(str(row.get("id")))),
            ("wrong_image", wrong_by_id.get(str(row.get("id")))),
        ]
        for mode, control_row in control_sources:
            if control_row is None:
                continue
            for rep in range(args.control_repeat):
                pairs.append(
                    pair(
                        control_row,
                        refusal_for(mode),
                        rejected_answer,
                        "control_refusal_over_hallucination",
                        mode,
                        f"rep{rep:02d}",
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
        "control_repeat": args.control_repeat,
        "normal_keep_ratio": args.normal_keep_ratio,
        "by_pair_type": dict(Counter(row["pair_type"] for row in pairs)),
        "by_train_input_mode": dict(Counter(row["train_input_mode"] for row in pairs)),
        "by_capability": dict(Counter(row["meta"]["capability"] for row in pairs)),
    }
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
