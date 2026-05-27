"""Audit model-mined Preference-v8.1 pairs before any DPO-v8.1 training."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.exclude_heldout_ids import identity_keys, read_heldout_keys, read_jsonl

CONTROL_TYPES = {
    "mined_text_prior_bias",
    "mined_blank_prior_answer",
    "mined_wrong_image_confound",
    "control_over_gold_overlap",
}
NORMAL_TYPE = "normal_anchor_gold_vs_model_wrong"
SPATIAL_TYPE = "spatial_gold_vs_spatial_wrong"
FORBIDDEN = re.compile(r"\b(?:image|visual|visible|scene|frame|picture)\b|图像|画面|图中|视觉", re.I)
REFUSAL = re.compile(r"insufficient evidence|does not support a reliable answer|cannot determine|unable to determine|无法判断|无法确定|缺少图像信息", re.I)
LEAK_FIELDS = {"reason", "references", "category", "subcategory", "debug", "metadata"}


def output_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"answer"} and isinstance(value.get("answer"), str)


def answer(pair: dict[str, Any], side: str) -> str:
    value = pair.get(side)
    return str(value.get("answer", "")) if isinstance(value, dict) else ""


def read_hard_lookup(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter[str]]:
    rows = read_jsonl(path)
    lookup = {(str(row["source_id"]), str(row["hard_type"])): row for row in rows}
    return lookup, Counter(str(row["hard_type"]) for row in rows)


def expected_rejected(pair: dict[str, Any], hard: dict[str, Any]) -> str:
    pair_type = str(pair.get("pair_type"))
    setting = "normal" if pair_type in {NORMAL_TYPE, SPATIAL_TYPE} else str(pair.get("setting"))
    return str(hard.get(f"{setting}_answer", ""))


def audit(pairs: list[dict[str, Any]], hard_lookup: dict[tuple[str, str], dict[str, Any]], available_hard: Counter[str], heldout_keys: set[str]) -> dict[str, Any]:
    total = len(pairs)
    by_type = Counter(str(pair.get("pair_type", "")) for pair in pairs)
    by_hard = Counter(str(pair.get("hard_type", "")) for pair in pairs)
    normal = [pair for pair in pairs if pair.get("pair_type") == NORMAL_TYPE]
    spatial = [pair for pair in pairs if pair.get("pair_type") == SPATIAL_TYPE]
    control = [pair for pair in pairs if pair.get("pair_type") in CONTROL_TYPES]
    valid_chosen = sum(output_valid(pair.get("chosen")) for pair in pairs)
    valid_rejected = sum(output_valid(pair.get("rejected")) for pair in pairs)
    reason_count = sum(int(isinstance(pair.get(side), dict) and "reason" in pair[side]) for pair in pairs for side in ("chosen", "rejected"))
    metadata_leak = sum(int(isinstance(pair.get(side), dict) and bool(set(pair[side]) & LEAK_FIELDS)) for pair in pairs for side in ("chosen", "rejected"))
    leaking = [
        str(pair.get("id"))
        for pair in pairs
        if identity_keys(pair.get("source_id"), {"metadata": pair.get("metadata") or {}}) & heldout_keys
    ]
    matched_actual = 0
    for pair in pairs:
        hard = hard_lookup.get((str(pair.get("source_id")), str(pair.get("hard_type"))))
        if hard and answer(pair, "rejected") == expected_rejected(pair, hard):
            matched_actual += 1
    duplicates = total - len({
        (str(pair.get("source_id")), str(pair.get("pair_type")), str(pair.get("setting")), answer(pair, "chosen"), answer(pair, "rejected"))
        for pair in pairs
    })
    chosen_valid_rate = valid_chosen / total if total else 0.0
    rejected_valid_rate = valid_rejected / total if total else 0.0
    normal_ratio = len(normal) / total if total else 0.0
    control_ratio = len(control) / total if total else 0.0
    spatial_ratio = len(spatial) / total if total else 0.0
    normal_refusal_rate = sum(bool(REFUSAL.search(answer(pair, "chosen"))) for pair in normal) / len(normal) if normal else 0.0
    control_forbidden_rate = sum(bool(FORBIDDEN.search(answer(pair, "chosen"))) for pair in control) / len(control) if control else 0.0
    duplicate_rate = duplicates / total if total else 0.0
    actual_rate = matched_actual / total if total else 0.0
    minimums = {
        "mined_text_prior_bias": ("text_prior_bias", 30),
        "mined_blank_prior_answer": ("blank_prior_answer", 30),
        "mined_wrong_image_confound": ("wrong_image_confound", 30),
    }
    fail_reasons = []
    checks = [
        (total > 0, "total pairs must be positive"),
        (not leaking, "held-out leakage count must equal 0"),
        (chosen_valid_rate > 0.98 and rejected_valid_rate > 0.98, "chosen/rejected JSON valid rate must exceed 98%"),
        (reason_count == 0, "reason field count must equal 0"),
        (metadata_leak == 0, "assistant metadata leakage count must equal 0"),
        (normal_ratio >= 0.25, "normal pair ratio must be at least 0.25"),
        (control_ratio <= 0.70, "control pair ratio must be at most 0.70"),
        (normal_refusal_rate < 0.02, "normal chosen refusal rate must be below 2%"),
        (control_forbidden_rate < 0.02, "control chosen forbidden visual terms rate must be below 2%"),
        (duplicate_rate < 0.15, "duplicate pair rate must be below 15%"),
        (actual_rate > 0.90, "rejected_from_actual_prediction rate must exceed 90%"),
    ]
    for passed, message in checks:
        if not passed:
            fail_reasons.append(message)
    minimum_status = {}
    for pair_type, (hard_type, minimum) in minimums.items():
        available = available_hard[hard_type]
        required = minimum if available >= minimum else available
        generated = by_type[pair_type]
        minimum_status[pair_type] = {"available": available, "required": required, "generated": generated, "passed": generated >= required}
        if generated < required:
            fail_reasons.append(f"{pair_type} requires {required} pairs when available, found {generated}")
    return {
        "total_pairs": total,
        "uses_model_predictions": True,
        "pair_type_counts": dict(by_type),
        "pair_type_ratios": {key: value / total if total else 0.0 for key, value in by_type.items()},
        "hard_type_counts": dict(by_hard),
        "normal_pair_ratio": normal_ratio,
        "control_pair_ratio": control_ratio,
        "spatial_pair_ratio": spatial_ratio,
        "chosen_json_valid_rate": chosen_valid_rate,
        "rejected_json_valid_rate": rejected_valid_rate,
        "reason_field_count": reason_count,
        "metadata_leak_count": metadata_leak,
        "heldout_leakage_count": len(leaking),
        "heldout_leakage_pair_ids": leaking,
        "control_chosen_forbidden_visual_terms_rate": control_forbidden_rate,
        "normal_chosen_refusal_rate": normal_refusal_rate,
        "duplicate_pair_rate": duplicate_rate,
        "rejected_from_r3_actual_prediction_rate": actual_rate,
        "average_chosen_length": sum(len(answer(pair, "chosen")) for pair in pairs) / total if total else 0.0,
        "average_rejected_length": sum(len(answer(pair, "rejected")) for pair in pairs) / total if total else 0.0,
        "minimum_hard_pair_checks": minimum_status,
        "train_ready": not fail_reasons,
        "fail_reasons": fail_reasons,
    }


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Preference-v8.1 Audit",
        "",
        f"- total_pairs: {report['total_pairs']}",
        f"- uses_model_predictions: {str(report['uses_model_predictions']).lower()}",
        f"- normal_pair_ratio: {report['normal_pair_ratio']:.2%}",
        f"- control_pair_ratio: {report['control_pair_ratio']:.2%}",
        f"- spatial_pair_ratio: {report['spatial_pair_ratio']:.2%}",
        f"- heldout_leakage_count: {report['heldout_leakage_count']}",
        f"- rejected_from_r3_actual_prediction_rate: {report['rejected_from_r3_actual_prediction_rate']:.2%}",
        f"- control_chosen_forbidden_visual_terms_rate: {report['control_chosen_forbidden_visual_terms_rate']:.2%}",
        f"- duplicate_pair_rate: {report['duplicate_pair_rate']:.2%}",
        f"- train_ready: {str(report['train_ready']).lower()}",
        "",
        "## Pair Types",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(report["pair_type_counts"].items()))
    if report["fail_reasons"]:
        lines += ["", "## Fail Reasons"] + [f"- {reason}" for reason in report["fail_reasons"]]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit model-mined Preference-v8.1 data.")
    parser.add_argument("--input", "--preference_file", dest="input", default="data/train/preference_v8_1/preference_v8_1_pairs.jsonl")
    parser.add_argument("--hard_negatives", default="data/mining/v8_1/hard_negatives_v8_1.jsonl")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_json", default="outputs/data_audit/preference_v8_1_audit.json")
    parser.add_argument("--output_md", default="outputs/data_audit/preference_v8_1_audit.md")
    parser.add_argument("--output_dir", default="", help="Compatibility alias controlling audit output paths.")
    parser.add_argument("--run", action="store_true", help="Accepted for orchestration compatibility; audit is offline only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        args.output_json = str(output_dir / "preference_v8_1_audit.json")
        args.output_md = str(output_dir / "preference_v8_1_audit.md")
    pairs = read_jsonl(Path(args.input))
    hard_lookup, available = read_hard_lookup(Path(args.hard_negatives))
    _, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    report = audit(pairs, hard_lookup, available, heldout_keys)
    write_report(report, Path(args.output_json), Path(args.output_md))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["train_ready"]:
        raise SystemExit("Preference-v8.1 audit failed; DPO-v8.1 GPU training is blocked")


if __name__ == "__main__":
    main()
