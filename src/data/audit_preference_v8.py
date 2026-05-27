"""Audit DPO-v8 preference data for leakage, format, and calibration risks."""
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

NORMAL_TYPES = {"normal_anchor_gold_vs_bad", "normal_gold_vs_refusal"}
CONTROL_TYPES = {"control_abstain_vs_hallucination", "wrong_image_caution_vs_confident_answer"}
SPATIAL_TYPE = "spatial_gold_vs_spatial_wrong"
FORBIDDEN_VISUAL_RE = re.compile(r"\b(?:image|visual|visible|scene|frame|picture)\b|\u56fe\u50cf|\u753b\u9762|\u56fe\u4e2d|\u89c6\u89c9", re.I)
REFUSAL_RE = re.compile(r"insufficient evidence|does not support a reliable answer|cannot determine|unable to determine|\u65e0\u6cd5\u5224\u65ad|\u65e0\u6cd5\u786e\u5b9a|\u7f3a\u5c11\u56fe\u50cf\u4fe1\u606f", re.I)
LEAK_FIELDS = {"reason", "references", "category", "subcategory", "debug", "metadata"}


def output_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"answer"} and isinstance(value.get("answer"), str)


def answer(pair: dict[str, Any], side: str) -> str:
    value = pair.get(side)
    return str(value.get("answer", "")) if isinstance(value, dict) else ""


def duplicate_key(pair: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(pair.get("source_id", "")),
        str(pair.get("pair_type", "")),
        str(pair.get("setting", "")),
        str(pair.get("prompt", "")),
        answer(pair, "chosen"),
        answer(pair, "rejected"),
    )


def audit(pairs: list[dict[str, Any]], heldout_keys: set[str]) -> dict[str, Any]:
    total = len(pairs)
    by_type = Counter(str(pair.get("pair_type", "")) for pair in pairs)
    by_setting = Counter(str(pair.get("setting", "")) for pair in pairs)
    normal = [pair for pair in pairs if pair.get("pair_type") in NORMAL_TYPES]
    control = [pair for pair in pairs if pair.get("pair_type") in CONTROL_TYPES]
    spatial = [pair for pair in pairs if pair.get("pair_type") == SPATIAL_TYPE]
    chosen_valid = sum(output_valid(pair.get("chosen")) for pair in pairs)
    rejected_valid = sum(output_valid(pair.get("rejected")) for pair in pairs)
    reason_count = sum(
        int(isinstance(pair.get(side), dict) and "reason" in pair[side])
        for pair in pairs
        for side in ("chosen", "rejected")
    )
    metadata_leak_count = sum(
        int(isinstance(pair.get(side), dict) and bool(set(pair[side]) & LEAK_FIELDS))
        for pair in pairs
        for side in ("chosen", "rejected")
    )
    leakage_pairs = [
        str(pair.get("id", ""))
        for pair in pairs
        if identity_keys(pair.get("source_id"), {"metadata": pair.get("metadata") or {}}) & heldout_keys
    ]
    chosen_forbidden_count = sum(bool(FORBIDDEN_VISUAL_RE.search(answer(pair, "chosen"))) for pair in pairs)
    control_forbidden_count = sum(bool(FORBIDDEN_VISUAL_RE.search(answer(pair, "chosen"))) for pair in control)
    normal_refusal_count = sum(bool(REFUSAL_RE.search(answer(pair, "chosen"))) for pair in normal)
    keys = [duplicate_key(pair) for pair in pairs]
    duplicate_count = len(keys) - len(set(keys))
    chosen_lengths = [len(answer(pair, "chosen")) for pair in pairs]
    rejected_lengths = [len(answer(pair, "rejected")) for pair in pairs]
    chosen_valid_rate = chosen_valid / total if total else 0.0
    rejected_valid_rate = rejected_valid / total if total else 0.0
    normal_ratio = len(normal) / total if total else 0.0
    control_ratio = len(control) / total if total else 0.0
    spatial_ratio = len(spatial) / total if total else 0.0
    normal_refusal_rate = normal_refusal_count / len(normal) if normal else 0.0
    control_forbidden_rate = control_forbidden_count / len(control) if control else 0.0
    duplicate_rate = duplicate_count / total if total else 0.0
    fail_reasons = []
    conditions = [
        (len(leakage_pairs) == 0, "held-out leakage count must equal 0"),
        (chosen_valid_rate > 0.98 and rejected_valid_rate > 0.98, "chosen/rejected JSON valid rate must exceed 98%"),
        (reason_count == 0, "reason field count must equal 0"),
        (metadata_leak_count == 0, "assistant metadata leakage count must equal 0"),
        (normal_ratio >= 0.50, "normal pair ratio must be at least 0.50"),
        (control_ratio <= 0.45, "control pair ratio must be at most 0.45"),
        (normal_refusal_rate < 0.02, "normal chosen refusal rate must be below 2%"),
        (control_forbidden_rate < 0.02, "control chosen forbidden visual terms rate must be below 2%"),
        (duplicate_rate < 0.10, "duplicate pair rate must be below 10%"),
    ]
    for passed, reason in conditions:
        if not passed:
            fail_reasons.append(reason)
    return {
        "total_pairs": total,
        "pair_type_counts": dict(by_type),
        "pair_type_ratios": {key: value / total if total else 0.0 for key, value in by_type.items()},
        "normal_pair_ratio": normal_ratio,
        "control_pair_ratio": control_ratio,
        "spatial_pair_ratio": spatial_ratio,
        "per_setting_pair_distribution": dict(by_setting),
        "chosen_json_valid_rate": chosen_valid_rate,
        "rejected_json_valid_rate": rejected_valid_rate,
        "reason_field_count": reason_count,
        "metadata_leak_count": metadata_leak_count,
        "heldout_leakage_count": len(leakage_pairs),
        "heldout_leakage_pair_ids": leakage_pairs,
        "chosen_forbidden_visual_terms_rate": chosen_forbidden_count / total if total else 0.0,
        "control_chosen_forbidden_visual_terms_rate": control_forbidden_rate,
        "normal_chosen_refusal_rate": normal_refusal_rate,
        "average_chosen_answer_length": sum(chosen_lengths) / total if total else 0.0,
        "average_rejected_answer_length": sum(rejected_lengths) / total if total else 0.0,
        "duplicate_pair_rate": duplicate_rate,
        "duplicate_pair_count": duplicate_count,
        "train_ready": not fail_reasons,
        "fail_reasons": fail_reasons,
    }


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Preference-v8 Audit",
        "",
        f"- total_pairs: {report['total_pairs']}",
        f"- normal_pair_ratio: {report['normal_pair_ratio']:.2%}",
        f"- control_pair_ratio: {report['control_pair_ratio']:.2%}",
        f"- spatial_pair_ratio: {report['spatial_pair_ratio']:.2%}",
        f"- heldout_leakage_count: {report['heldout_leakage_count']}",
        f"- chosen_json_valid_rate: {report['chosen_json_valid_rate']:.2%}",
        f"- rejected_json_valid_rate: {report['rejected_json_valid_rate']:.2%}",
        f"- reason_field_count: {report['reason_field_count']}",
        f"- metadata_leak_count: {report['metadata_leak_count']}",
        f"- control_chosen_forbidden_visual_terms_rate: {report['control_chosen_forbidden_visual_terms_rate']:.2%}",
        f"- normal_chosen_refusal_rate: {report['normal_chosen_refusal_rate']:.2%}",
        f"- duplicate_pair_rate: {report['duplicate_pair_rate']:.2%}",
        f"- train_ready: {str(report['train_ready']).lower()}",
        "",
        "## Pair Types",
    ]
    for pair_type, count in report["pair_type_counts"].items():
        lines.append(f"- {pair_type}: {count} ({report['pair_type_ratios'][pair_type]:.2%})")
    if report["fail_reasons"]:
        lines += ["", "## Fail Reasons", *[f"- {reason}" for reason in report["fail_reasons"]]]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit answer-only Preference-v8 pairs.")
    parser.add_argument("--input", default="data/train/preference_v8/preference_v8_pairs.jsonl")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_json", default="outputs/data_audit/preference_v8_audit.json")
    parser.add_argument("--output_md", default="outputs/data_audit/preference_v8_audit.md")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(Path(args.input))
    heldout_ids, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    report = audit(pairs, heldout_keys)
    report["heldout_id_count_used_for_exclusion_only"] = len(heldout_ids)
    write_report(report, Path(args.output_json), Path(args.output_md))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["train_ready"]:
        raise SystemExit("Preference-v8 audit failed; DPO-v8 GPU training is blocked")


if __name__ == "__main__":
    main()
