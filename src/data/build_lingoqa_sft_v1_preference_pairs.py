"""Build LingoQA SFT-v1 preference pairs for strict visual grounding.

The goal is not to make a larger ordinary SFT set.  It is to create direct
preference signals for the failure mode seen in SFT-v1:

1. With the correct visual frames, a grounded visual answer should be preferred
   over a generic refusal.
2. With text-only, blank-image, or wrong-image controls, an uncertainty/refusal
   answer should be preferred over the original visually grounded answer.

The output JSONL is consumed by ``src/train_qwen25vl_dpo_lora.py``.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


REFUSAL = {
    "task": "external_vqa",
    "answer": "I cannot determine this from the provided visual input.",
    "reason": "The available visual input does not provide reliable evidence for this answer, so I should not guess.",
}


def read_jsonl(path: Path, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid jsonl") from exc
    return rows


def sample_id(row: dict[str, Any]) -> str:
    return str(row.get("id", ""))


def answer_obj(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer", {})
    if isinstance(answer, dict):
        task = answer.get("task") or row.get("meta", {}).get("task_type") or "external_vqa"
        return {
            "task": task,
            "answer": str(answer.get("answer", "")),
            "reason": str(answer.get("reason", "")),
        }
    return {"task": row.get("meta", {}).get("task_type", "external_vqa"), "answer": str(answer), "reason": ""}


def refusal_for(row: dict[str, Any], mode: str) -> dict[str, str]:
    if mode == "text_only":
        reason = "No image frames are available, so the visual question cannot be answered reliably."
    elif mode == "blank_image":
        reason = "The image frames are blank placeholders and do not contain the requested driving-scene evidence."
    elif mode == "wrong_image":
        reason = "The provided frames do not match the question's original scene, so the visually grounded answer is not reliable."
    else:
        reason = REFUSAL["reason"]
    return {"task": "external_vqa", "answer": REFUSAL["answer"], "reason": reason}


def train_mode(row: dict[str, Any]) -> str:
    meta = row.get("meta", {})
    for key in ("train_input_mode", "control_type", "visual_control_type"):
        value = meta.get(key)
        if value:
            return str(value)
    review = row.get("review", {})
    value = review.get("train_input_mode") or review.get("control_type")
    if value:
        return str(value)
    return "normal"


def original_ids(row: dict[str, Any]) -> list[str]:
    meta = row.get("meta", {})
    review = row.get("review", {})
    candidates = [
        review.get("original_id"),
        review.get("source_id"),
        meta.get("original_id"),
        meta.get("source_id"),
        meta.get("base_id"),
    ]
    row_id = sample_id(row)
    if row_id.endswith("_text_only_control"):
        candidates.append(row_id.replace("_text_only_control", ""))
    if row_id.endswith("_blank_control"):
        candidates.append(row_id.replace("_blank_control", ""))
    if row_id.endswith("_wrong_image_control"):
        candidates.append(row_id.replace("_wrong_image_control", ""))
    return [str(item) for item in candidates if item]


def make_prompt_sample(row: dict[str, Any]) -> dict[str, Any]:
    sample = copy.deepcopy(row)
    sample.pop("answer", None)
    return sample


def pair_record(
    row: dict[str, Any],
    chosen: dict[str, Any],
    rejected: dict[str, Any],
    pair_type: str,
    mode: str,
    source: str,
) -> dict[str, Any]:
    return {
        "id": f"{sample_id(row)}__{pair_type}",
        "source_id": sample_id(row),
        "pair_type": pair_type,
        "train_input_mode": mode,
        "prompt_sample": make_prompt_sample(row),
        "chosen": chosen,
        "rejected": rejected,
        "meta": {
            "source": "lingoqa_sft_v1_preference",
            "pair_source": source,
            "capability": row.get("meta", {}).get("capability")
            or row.get("answer", {}).get("subcategory")
            or row.get("perception", {}).get("risk_hint", ""),
            "sft_v1_role": row.get("meta", {}).get("sft_v1_role", ""),
        },
    }


def build_pairs(rows: list[dict[str, Any]], lookup_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter[str]]:
    by_id = {sample_id(row): row for row in lookup_rows if sample_id(row)}
    by_id.update({sample_id(row): row for row in rows if sample_id(row)})

    pairs: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for row in rows:
        mode = train_mode(row)
        row_id = sample_id(row)
        if not row_id:
            skipped["missing_id"] += 1
            continue

        if mode == "normal":
            chosen = answer_obj(row)
            if not chosen.get("answer"):
                skipped["normal_missing_answer"] += 1
                continue
            pairs.append(
                pair_record(
                    row=row,
                    chosen=chosen,
                    rejected=REFUSAL,
                    pair_type="normal_answer_over_refusal",
                    mode=mode,
                    source=row_id,
                )
            )
            continue

        if mode not in {"text_only", "blank_image", "wrong_image"}:
            skipped[f"unknown_mode:{mode}"] += 1
            continue

        original = None
        for oid in original_ids(row):
            if oid in by_id:
                original = by_id[oid]
                break
        if original is None:
            skipped[f"{mode}_missing_original"] += 1
            continue

        rejected = answer_obj(original)
        if not rejected.get("answer"):
            skipped[f"{mode}_missing_rejected_answer"] += 1
            continue
        chosen = answer_obj(row)
        if not chosen.get("answer") or "cannot determine" not in chosen.get("answer", "").lower():
            chosen = refusal_for(row, mode)

        pairs.append(
            pair_record(
                row=row,
                chosen=chosen,
                rejected=rejected,
                pair_type="control_refusal_over_hallucination",
                mode=mode,
                source=sample_id(original),
            )
        )

    return pairs, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LingoQA SFT-v1 DPO preference pairs.")
    parser.add_argument("--input", default="data/processed/lingoqa_sft_v1_control_aware_77.jsonl")
    parser.add_argument(
        "--original_lookup",
        default="data/processed/lingoqa_sft_v1_balanced_67.jsonl",
        help="Optional normal-answer lookup used by control rows whose source example was removed from ordinary SFT.",
    )
    parser.add_argument("--output", default="data/processed/lingoqa_sft_v1_preference_pairs_77.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_sft_v1_preference_pairs_77_summary.json")
    args = parser.parse_args()

    input_path = ROOT / args.input
    lookup_path = ROOT / args.original_lookup
    output_path = ROOT / args.output
    summary_path = ROOT / args.summary

    rows = read_jsonl(input_path)
    lookup_rows = read_jsonl(lookup_path, required=False)
    pairs, skipped = build_pairs(rows, lookup_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    by_pair_type = Counter(pair["pair_type"] for pair in pairs)
    by_mode = Counter(pair["train_input_mode"] for pair in pairs)
    by_capability = Counter(pair.get("meta", {}).get("capability", "") for pair in pairs)
    summary = {
        "input": str(input_path.relative_to(ROOT)),
        "original_lookup": str(lookup_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "input_rows": len(rows),
        "lookup_rows": len(lookup_rows),
        "preference_pairs": len(pairs),
        "by_pair_type": dict(by_pair_type),
        "by_train_input_mode": dict(by_mode),
        "by_capability": dict(by_capability),
        "skipped": dict(skipped),
        "recommendation": (
            "Use DPO as the next optimization step. Ordinary SFT has already shown that it can raise "
            "normal F1 while also raising wrong-image controls; these pairs directly penalize that behavior."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
