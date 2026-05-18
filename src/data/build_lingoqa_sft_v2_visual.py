"""Build a larger clean LingoQA SFT-v2 visual training set.

The source should be a clean train split produced by
``build_lingoqa_clean_splits.py``. The script expands each sample by reference
answer variants and applies conservative capability repeats so rare capabilities
are not drowned by spatial-localization rows.
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

DEFAULT_REPEAT = {
    "spatial_localization": 1,
    "other": 1,
    "counting": 2,
    "object_recognition": 2,
    "reasoning_world_knowledge": 4,
    "weather_road_condition": 4,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def capability(row: dict[str, Any]) -> str:
    return str(
        row.get("meta", {}).get("external", {}).get("capability")
        or row.get("answer", {}).get("subcategory")
        or row.get("perception", {}).get("risk_hint")
        or "unknown"
    )


def parse_repeats(text: str) -> dict[str, int]:
    repeats = dict(DEFAULT_REPEAT)
    if not text:
        return repeats
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        repeats[key.strip()] = max(1, int(value))
    return repeats


def references(row: dict[str, Any], mode: str) -> list[str]:
    answer = row.get("answer", {})
    refs: list[str] = []
    if isinstance(answer, dict):
        primary = str(answer.get("answer", "")).strip()
        if primary:
            refs.append(primary)
        if mode == "all":
            for ref in answer.get("references", []):
                ref_text = str(ref).strip()
                if ref_text and ref_text not in refs:
                    refs.append(ref_text)
    return refs[:1] if mode == "primary" else refs


def build_rows(rows: list[dict[str, Any]], reference_mode: str, repeats: dict[str, int], seed: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        cap = capability(row)
        row_refs = references(row, reference_mode) or [str(row.get("answer", {}).get("answer", ""))]
        for ref_idx, ref in enumerate(row_refs):
            for rep_idx in range(repeats.get(cap, 1)):
                sample = copy.deepcopy(row)
                sample["id"] = f"{row.get('id')}__sftv2_ref{ref_idx:02d}_rep{rep_idx:02d}"
                sample.setdefault("answer", {})
                sample["answer"]["answer"] = ref
                sample["answer"]["reason"] = ref
                sample["answer"]["references"] = row_refs
                sample.setdefault("meta", {})
                sample["meta"]["source"] = "lingoqa_sft_v2_visual"
                sample["meta"]["sft_v2_role"] = "normal_visual_sft"
                sample["meta"]["capability_repeat"] = repeats.get(cap, 1)
                sample["meta"]["reference_variant_index"] = ref_idx
                out.append(sample)
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean LingoQA SFT-v2 visual data.")
    parser.add_argument("--input", default="data/processed/lingoqa_clean_v2_train.jsonl")
    parser.add_argument("--output", default="data/processed/lingoqa_sft_v2_visual_train.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_sft_v2_visual_train_summary.json")
    parser.add_argument("--reference_variants", choices=["primary", "all"], default="all")
    parser.add_argument("--capability_repeats", default="")
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    source_rows = read_jsonl(ROOT / args.input)
    repeats = parse_repeats(args.capability_repeats)
    train_rows = build_rows(source_rows, args.reference_variants, repeats, args.seed)
    write_jsonl(ROOT / args.output, train_rows)

    summary = {
        "input": args.input,
        "output": args.output,
        "source_rows": len(source_rows),
        "train_rows": len(train_rows),
        "reference_variants": args.reference_variants,
        "capability_repeats": repeats,
        "source_by_capability": dict(Counter(capability(row) for row in source_rows)),
        "train_by_capability": dict(Counter(capability(row) for row in train_rows)),
    }
    summary_path = ROOT / args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
