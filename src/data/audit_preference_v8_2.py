"""Audit case-gap-aware Preference-v8.2 pairs before GPU training."""
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
    "control_direct_answer_vs_caution",
    "blank_high_f1_vs_caution",
    "text_only_gold_overlap_vs_caution",
    "wrong_image_gold_overlap_vs_caution",
}
NORMAL_TYPES = {"normal_anchor_gold_vs_model_wrong", "normal_gold_vs_refusal"}
SPATIAL_TYPE = "spatial_gold_vs_spatial_wrong"
FORBIDDEN = re.compile(r"\b(?:image|visual|visible|scene|frame|picture)\b|图像|画面|图中|视觉", re.I)
REFUSAL = re.compile(r"insufficient evidence|does not support a reliable answer|cannot determine|unable to determine|无法判断|不能确定|缺少信息|信息不足", re.I)
LEAK_FIELDS = {"reason", "references", "category", "subcategory", "debug", "metadata"}


def output_valid(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"answer"} and isinstance(value.get("answer"), str)


def answer(pair: dict[str, Any], side: str) -> str:
    value = pair.get(side)
    return str(value.get("answer", "")) if isinstance(value, dict) else ""


def audit(pairs: list[dict[str, Any]], candidates: list[dict[str, Any]], heldout_keys: set[str], dry_run: bool = False) -> dict[str, Any]:
    total = len(pairs)
    counts = Counter(str(pair.get("pair_type", "")) for pair in pairs)
    normal = [pair for pair in pairs if pair.get("pair_type") in NORMAL_TYPES]
    control = [pair for pair in pairs if pair.get("pair_type") in CONTROL_TYPES]
    spatial = [pair for pair in pairs if pair.get("pair_type") == SPATIAL_TYPE]
    candidate_counts = Counter(str(row.get("candidate_type", "")) for row in candidates)
    leaking = [str(pair.get("id", "")) for pair in pairs if identity_keys(pair.get("source_id"), {"metadata": pair.get("metadata") or {}}) & heldout_keys]
    chosen_valid = sum(output_valid(pair.get("chosen")) for pair in pairs)
    rejected_valid = sum(output_valid(pair.get("rejected")) for pair in pairs)
    reason_count = sum(int(isinstance(pair.get(side), dict) and "reason" in pair[side]) for pair in pairs for side in ("chosen", "rejected"))
    metadata_leak = sum(int(isinstance(pair.get(side), dict) and bool(set(pair[side]) & LEAK_FIELDS)) for pair in pairs for side in ("chosen", "rejected"))
    duplicate_count = total - len({
        (str(pair.get("source_id")), str(pair.get("setting")), answer(pair, "chosen"), answer(pair, "rejected"))
        for pair in pairs
    })
    candidate_lookup = {
        (str(row.get("source_id", "")), str(row.get("candidate_type", "")), str(row.get("setting", ""))): row
        for row in candidates
    }
    actual_count = 0
    for pair in pairs:
        metadata = pair.get("metadata") or {}
        candidate = candidate_lookup.get((str(pair.get("source_id", "")), str(metadata.get("candidate_type", "")), str(pair.get("setting", ""))))
        if (
            bool(metadata.get("rejected_from_actual_prediction"))
            and candidate
            and bool(candidate.get("rejected_from_actual_prediction"))
            and answer(pair, "rejected") == str(candidate.get("model_answer", ""))
        ):
            actual_count += 1
    rates = {
        "chosen_json_valid_rate": chosen_valid / total if total else 0.0,
        "rejected_json_valid_rate": rejected_valid / total if total else 0.0,
        "normal_pair_ratio": len(normal) / total if total else 0.0,
        "control_pair_ratio": len(control) / total if total else 0.0,
        "spatial_pair_ratio": len(spatial) / total if total else 0.0,
        "control_chosen_forbidden_visual_terms_rate": sum(bool(FORBIDDEN.search(answer(pair, "chosen"))) for pair in control) / len(control) if control else 0.0,
        "normal_chosen_refusal_rate": sum(bool(REFUSAL.search(answer(pair, "chosen"))) for pair in normal) / len(normal) if normal else 0.0,
        "duplicate_pair_rate": duplicate_count / total if total else 0.0,
        "rejected_from_actual_prediction_rate": actual_count / total if total else 0.0,
    }
    fail_reasons: list[str] = []
    checks = [
        (total > 0, "total_pairs must be positive"),
        (not leaking, "heldout_leakage_count must equal 0"),
        (rates["chosen_json_valid_rate"] > 0.98 and rates["rejected_json_valid_rate"] > 0.98, "chosen/rejected JSON valid rate must exceed 98%"),
        (reason_count == 0, "reason_field_count must equal 0"),
        (metadata_leak == 0, "metadata_leak_count must equal 0"),
        (rates["normal_chosen_refusal_rate"] < 0.02, "normal chosen refusal rate must be below 2%"),
        (rates["control_chosen_forbidden_visual_terms_rate"] < 0.02, "control chosen forbidden visual terms rate must be below 2%"),
        (rates["duplicate_pair_rate"] < 0.15, "duplicate pair rate must be below 15%"),
    ]
    if not dry_run:
        checks += [
            (rates["normal_pair_ratio"] >= 0.35, "normal pair ratio must be at least 0.35"),
            (rates["control_pair_ratio"] <= 0.60, "control pair ratio must be at most 0.60"),
            (rates["spatial_pair_ratio"] >= 0.03, "spatial pair ratio must be at least 0.03"),
            (rates["rejected_from_actual_prediction_rate"] >= 0.80, "rejected_from_actual_prediction rate must be at least 80%"),
        ]
    for passed, reason in checks:
        if not passed:
            fail_reasons.append(reason)
    minimums = [
        ("blank_high_f1_vs_caution", "blank_high_f1_prior", 30),
        ("control_direct_answer_vs_caution", "control_direct_answer", 50),
    ]
    warnings: list[str] = []
    for pair_type, candidate_type, required_if_available in minimums:
        available = candidate_counts[candidate_type]
        required = required_if_available if available >= required_if_available else available
        if not dry_run and counts[pair_type] < required:
            fail_reasons.append(f"{pair_type} must include {required} pairs when available, found {counts[pair_type]}")
        if available < required_if_available:
            warnings.append(f"Only {available} candidates are available for {pair_type}; reduced minimum applied.")
    return {
        "total_pairs": total,
        "dry_run": dry_run,
        "uses_model_predictions": True,
        "pair_type_counts": dict(counts),
        "pair_type_ratios": {key: value / total if total else 0.0 for key, value in sorted(counts.items())},
        **rates,
        "chosen_json_valid_rate": rates["chosen_json_valid_rate"],
        "rejected_json_valid_rate": rates["rejected_json_valid_rate"],
        "reason_field_count": reason_count,
        "metadata_leak_count": metadata_leak,
        "heldout_leakage_count": len(leaking),
        "heldout_leakage_pair_ids": leaking,
        "blank_high_f1_pair_count": counts["blank_high_f1_vs_caution"],
        "text_only_gold_overlap_pair_count": counts["text_only_gold_overlap_vs_caution"],
        "wrong_image_gold_overlap_pair_count": counts["wrong_image_gold_overlap_vs_caution"],
        "control_direct_answer_pair_count": counts["control_direct_answer_vs_caution"],
        "avg_chosen_length": sum(len(answer(pair, "chosen")) for pair in pairs) / total if total else 0.0,
        "avg_rejected_length": sum(len(answer(pair, "rejected")) for pair in pairs) / total if total else 0.0,
        "train_ready": not fail_reasons,
        "fail_reasons": fail_reasons,
        "warnings": warnings,
    }


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Preference-v8.2 Audit",
        "",
        "This audit is CPU-only and does not train or evaluate any model.",
        "",
        f"- total_pairs: {report['total_pairs']}",
        f"- normal_pair_ratio: {report['normal_pair_ratio']:.2%}",
        f"- control_pair_ratio: {report['control_pair_ratio']:.2%}",
        f"- spatial_pair_ratio: {report['spatial_pair_ratio']:.2%}",
        f"- heldout_leakage_count: {report['heldout_leakage_count']}",
        f"- rejected_from_actual_prediction_rate: {report['rejected_from_actual_prediction_rate']:.2%}",
        f"- control_chosen_forbidden_visual_terms_rate: {report['control_chosen_forbidden_visual_terms_rate']:.2%}",
        f"- duplicate_pair_rate: {report['duplicate_pair_rate']:.2%}",
        f"- train_ready: {str(report['train_ready']).lower()}",
        "",
        "## Pair Types",
    ]
    lines.extend(f"- {key}: {value} ({report['pair_type_ratios'][key]:.2%})" for key, value in sorted(report["pair_type_counts"].items()))
    if report["warnings"]:
        lines += ["", "## Warnings"] + [f"- {item}" for item in report["warnings"]]
    if report["fail_reasons"]:
        lines += ["", "## Fail Reasons"] + [f"- {item}" for item in report["fail_reasons"]]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Preference-v8.2 before DPO-v8.2 smoke.")
    parser.add_argument("--preference_file", default="data/train/preference_v8_2/preference_v8_2_pairs.jsonl")
    parser.add_argument("--candidates", default="data/mining/v8_2/preference_v8_2_candidate_cases.jsonl")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_dir", default="outputs/data_audit")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = read_jsonl(Path(args.preference_file))
    candidates = read_jsonl(Path(args.candidates))
    _, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    report = audit(pairs, candidates, heldout_keys, dry_run=args.dry_run)
    output_dir = Path(args.output_dir)
    write_report(report, output_dir / "preference_v8_2_audit.json", output_dir / "preference_v8_2_audit.md")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["train_ready"]:
        raise SystemExit("Preference-v8.2 audit failed; DPO-v8.2 GPU smoke is blocked")


if __name__ == "__main__":
    main()
