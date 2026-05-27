"""Build mixed normal/control SFT rows for DriveLM visual dependency.

Normal samples keep evidence-aware gold answers. Control samples (text-only,
hard wrong-image, and blank-image) are supervised to an uncertainty answer.
This is stronger than DPO for the observed failure mode where the model keeps
answering ablated inputs from language priors.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


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


def copy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    return str(answer.get("answer") or answer.get("reason") or "").strip()


def uncertainty_answer(mode: str) -> dict[str, str]:
    if mode == "text_only":
        reason = "No visual frames are available, so this scene-specific driving question cannot be answered reliably."
    elif mode == "blank_image":
        reason = "The provided frames contain no usable driving-scene evidence, so a scene-specific answer would be unreliable."
    elif mode == "wrong_image":
        reason = "The provided frames are not reliable evidence for the queried driving scene."
    else:
        reason = "The visual evidence is insufficient to answer this scene-specific driving question reliably."
    return {
        "task": "external_vqa",
        "answer": "I cannot determine this from the provided visual input.",
        "reason": reason,
    }


def capability(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    return str(meta.get("capability") or external.get("capability") or answer.get("subcategory") or "unknown")


def limited_shuffle(rows: list[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    out = list(rows)
    random.Random(seed).shuffle(out)
    if limit > 0:
        out = out[:limit]
    return out


def normalize_row(row: dict[str, Any], mode: str, answer: dict[str, Any], recipe: str) -> dict[str, Any]:
    new_row = copy_json(row)
    new_row["answer"] = copy_json(answer)
    meta = new_row.setdefault("meta", {})
    meta["train_input_mode"] = mode
    meta["control_sft_version"] = recipe
    meta["recommended_perception_mode"] = "none"
    return new_row


def add_normal_rows(output: list[dict[str, Any]], rows: list[dict[str, Any]], counters: Counter[str], recipe: str) -> None:
    for row in rows:
        if not answer_text(row):
            counters["normal_skipped_empty_answer"] += 1
            continue
        output.append(normalize_row(row, "normal", row["answer"], recipe))
        counters["normal_added"] += 1


def add_control_rows(
    output: list[dict[str, Any]],
    *,
    mode: str,
    base_rows: list[dict[str, Any]],
    control_by_id: dict[str, dict[str, Any]],
    max_per_capability: int,
    counters: Counter[str],
    recipe: str,
) -> None:
    per_cap: Counter[str] = Counter()
    for base_row in base_rows:
        source_id = str(base_row.get("id"))
        row = base_row if mode == "text_only" else control_by_id.get(source_id)
        if row is None:
            counters[f"{mode}_skipped_missing_control"] += 1
            continue
        cap = capability(base_row)
        if max_per_capability > 0 and per_cap[cap] >= max_per_capability:
            counters[f"{mode}_skipped_capability_quota"] += 1
            continue
        output.append(normalize_row(row, mode, uncertainty_answer(mode), recipe))
        per_cap[cap] += 1
        counters[f"{mode}_added"] += 1


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normal_rows_all = read_jsonl(Path(args.normal_data))
    wrong_by_id = index_by_id(read_jsonl(Path(args.wrong_data)))
    blank_by_id = index_by_id(read_jsonl(Path(args.blank_data)))
    normal_rows = limited_shuffle(normal_rows_all, args.max_normal_rows, args.seed)
    control_base_rows = limited_shuffle(normal_rows_all, args.max_control_rows_per_mode, args.seed + 29)
    enabled_modes = {item.strip() for item in args.control_modes.split(",") if item.strip()}

    output: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    recipe = "drivelm_control_sft_v1"
    add_normal_rows(output, normal_rows, counters, recipe)
    if "text_only" in enabled_modes:
        add_control_rows(
            output,
            mode="text_only",
            base_rows=control_base_rows,
            control_by_id={},
            max_per_capability=args.max_control_rows_per_capability,
            counters=counters,
            recipe=recipe,
        )
    if "wrong_image" in enabled_modes:
        add_control_rows(
            output,
            mode="wrong_image",
            base_rows=control_base_rows,
            control_by_id=wrong_by_id,
            max_per_capability=args.max_control_rows_per_capability,
            counters=counters,
            recipe=recipe,
        )
    if "blank_image" in enabled_modes:
        add_control_rows(
            output,
            mode="blank_image",
            base_rows=control_base_rows,
            control_by_id=blank_by_id,
            max_per_capability=args.max_control_rows_per_capability,
            counters=counters,
            recipe=recipe,
        )

    random.Random(args.seed + 101).shuffle(output)
    by_mode = Counter(str(row.get("meta", {}).get("train_input_mode")) for row in output)
    by_capability = Counter(capability(row) for row in output)
    summary = {
        "recipe": recipe,
        "rows": len(output),
        "normal_rows": len(normal_rows_all),
        "control_modes": sorted(enabled_modes),
        "limits": {
            "max_normal_rows": args.max_normal_rows,
            "max_control_rows_per_mode": args.max_control_rows_per_mode,
            "max_control_rows_per_capability": args.max_control_rows_per_capability,
        },
        "by_train_input_mode": dict(by_mode.most_common()),
        "by_capability": dict(by_capability.most_common()),
        "counters": dict(counters.most_common()),
        "recommended_training_args": {
            "init_adapter_path": "outputs/checkpoints/qwen25vl_3b_drivelm_evidence_sft_v1",
            "prompt_variant": "evidence",
            "perception_mode": "none",
            "learning_rate": "6e-7 to 1e-6",
        },
        "inputs": {
            "normal_data": args.normal_data,
            "wrong_data": args.wrong_data,
            "blank_data": args.blank_data,
        },
    }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM mixed normal/control SFT data.")
    parser.add_argument("--normal_data", default="data/processed/drivelm_evidence_sft_v1_train.jsonl")
    parser.add_argument("--wrong_data", default="data/processed/drivelm_train_scene_wrong_image_hard.jsonl")
    parser.add_argument("--blank_data", default="data/processed/drivelm_train_scene_blank_image_v4.jsonl")
    parser.add_argument("--output", default="data/processed/drivelm_control_sft_v1_train.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/drivelm_control_sft_v1_train_summary.json")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--control_modes", default="text_only,wrong_image,blank_image")
    parser.add_argument("--max_normal_rows", type=int, default=1200)
    parser.add_argument("--max_control_rows_per_mode", type=int, default=700)
    parser.add_argument("--max_control_rows_per_capability", type=int, default=180)
    args = parser.parse_args()

    rows, summary = build_rows(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output} and {args.summary}")


if __name__ == "__main__":
    main()
