"""Collect non-heldout, case-gap-aware candidates for Preference-v8.2.

Stage 8.5 artifacts are read only as diagnostic evidence: they are based on
held-out evaluation and never become training pair sources. Actual candidate
answers come from the non-heldout r3 mining pool and normal r3 SFT anchors.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.exclude_heldout_ids import identity_keys, read_heldout_keys, read_jsonl, write_jsonl
from src.eval.control_behavior_metrics import behavior_flags, is_refusal_or_caution

CONTROL_SETTINGS = ("text_only", "wrong_image", "blank_image")
SPATIAL_TERMS = ("left", "right", "front", "back", "lane", "traffic light", "pedestrian", "vehicle", "cyclist", "car")
NEUTRAL_ABSTAIN = "Insufficient evidence to answer reliably."


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_diagnostic_counts(paths: list[Path]) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    total_rows = 0
    ids: set[str] = set()
    for path in paths:
        if not path.exists():
            detail[str(path)] = {"exists": False, "rows": 0}
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        row_ids = {str(row.get("id", "")) for row in rows if row.get("id")}
        detail[str(path)] = {"exists": True, "rows": len(rows), "unique_ids": len(row_ids)}
        total_rows += len(rows)
        ids.update(row_ids)
    return {
        "files": detail,
        "rows_read_for_diagnosis_only": total_rows,
        "unique_ids_read_for_diagnosis_only": len(ids),
        "training_candidates_from_diagnostic_files": 0,
        "note": "Held-out Stage 8.5 records inform the design only; they are excluded from pair construction.",
    }


def consolidate_hard_rows(rows: list[dict[str, Any]], heldout_keys: set[str]) -> tuple[dict[str, dict[str, Any]], int]:
    cases: dict[str, dict[str, Any]] = {}
    removed = 0
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if not source_id or identity_keys(source_id) & heldout_keys:
            removed += 1
            continue
        case = cases.setdefault(source_id, dict(row))
        case.setdefault("_hard_types", set()).add(str(row.get("hard_type", "")))
    return cases, removed


def make_candidate(case: dict[str, Any], candidate_type: str, setting: str, trigger: str, actual: bool = True) -> dict[str, Any]:
    source_id = str(case["source_id"])
    answer = str(case.get(f"{setting}_answer", "")) if setting in CONTROL_SETTINGS or setting == "normal" else ""
    control_f1 = float(case.get(f"{setting}_f1", 0.0)) if setting in CONTROL_SETTINGS else 0.0
    flags = behavior_flags(answer, control_f1)
    return {
        "id": f"{source_id}_{candidate_type}_{setting}",
        "source_id": source_id,
        "candidate_type": candidate_type,
        "setting": setting,
        "question": str(case.get("question", "")),
        "gold": str(case.get("gold", "")),
        "model_answer": answer,
        "normal_answer": str(case.get("normal_answer", "")),
        "text_only_answer": str(case.get("text_only_answer", "")),
        "wrong_image_answer": str(case.get("wrong_image_answer", "")),
        "blank_image_answer": str(case.get("blank_image_answer", "")),
        "normal_f1": float(case.get("normal_f1", 0.0)),
        "text_only_f1": float(case.get("text_only_f1", 0.0)),
        "wrong_image_f1": float(case.get("wrong_image_f1", 0.0)),
        "blank_image_f1": float(case.get("blank_image_f1", 0.0)),
        "control_f1": control_f1,
        "case_gap": float(case.get("case_gap", 0.0)),
        "is_direct_answer": flags["is_direct_answer"],
        "is_short_prior_answer": flags["is_short_prior_answer"],
        "is_caution": flags["is_caution"],
        "source_file": "data/mining/v8_1/hard_negatives_v8_1.jsonl",
        "heldout_excluded": True,
        "rejected_from_actual_prediction": actual,
        "trigger_reason": trigger,
    }


def collect_mined_candidates(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_id, case in cases.items():
        hard_types = case.get("_hard_types", set())
        nf = float(case.get("normal_f1", 0.0))
        for setting in CONTROL_SETTINGS:
            f1 = float(case.get(f"{setting}_f1", 0.0))
            answer = str(case.get(f"{setting}_answer", ""))
            flags = behavior_flags(answer, f1)
            if setting == "blank_image" and f1 >= 0.20 and not flags["is_caution"]:
                candidates.append(make_candidate(case, "blank_high_f1_prior", setting, "blank_image answer has F1 >= 0.20 without caution"))
            if setting == "text_only" and f1 >= 0.20 and not flags["is_caution"] and (flags["is_direct_answer"] or flags["is_short_prior_answer"]):
                candidates.append(make_candidate(case, "text_only_gold_overlap", setting, "text_only direct/prior answer has F1 >= 0.20"))
            if setting == "wrong_image" and not flags["is_caution"] and (f1 >= 0.20 or f1 >= nf - 0.05):
                candidates.append(make_candidate(case, "wrong_image_gold_overlap", setting, "wrong_image answer overlaps gold or nearly matches normal F1"))
            if not flags["is_caution"] and (flags["is_direct_answer"] or flags["is_short_prior_answer"]):
                candidates.append(make_candidate(case, "control_direct_answer", setting, "control output is a direct or short-prior answer"))
        max_control = max(float(case.get(f"{setting}_f1", 0.0)) for setting in CONTROL_SETTINGS)
        if "control_over_gold_overlap" in hard_types or max_control >= nf or max_control >= 0.30:
            best = max(CONTROL_SETTINGS, key=lambda setting: float(case.get(f"{setting}_f1", 0.0)))
            candidates.append(make_candidate(case, "control_over_gold_overlap", best, "best control F1 reaches normal F1 or high-overlap threshold"))
        if ("normal_wrong_anchor" in hard_types or nf < 0.25) and str(case.get("gold", "")).strip() and not is_refusal_or_caution(str(case.get("gold", ""))):
            candidates.append(make_candidate(case, "normal_model_wrong", "normal", "normal model answer has low F1; preserve gold anchor"))
        question = str(case.get("question", "")).lower()
        if ("spatial_relation_error" in hard_types or any(term in question for term in SPATIAL_TERMS)) and nf < 0.30 and str(case.get("gold", "")).strip():
            candidates.append(make_candidate(case, "spatial_relation_error", "normal", "spatial question with low normal F1"))
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in candidates:
        unique.setdefault((row["source_id"], row["candidate_type"], row["setting"]), row)
    return list(unique.values())


def collect_refusal_anchors(sft_rows: list[dict[str, Any]], heldout_keys: set[str]) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    removed = 0
    seen: set[str] = set()
    for row in sft_rows:
        if str(row.get("sample_type", "")) not in {"normal_replay", "spatial_normal_qa", "extra_normal_fill"}:
            continue
        metadata = row.get("metadata") or {}
        source_id = str(metadata.get("source_id") or row.get("id", ""))
        if not source_id or source_id in seen:
            continue
        if identity_keys(source_id, row) & heldout_keys:
            removed += 1
            continue
        gold = str((row.get("assistant") or {}).get("answer", ""))
        if not gold or is_refusal_or_caution(gold):
            continue
        seen.add(source_id)
        candidates.append({
            "id": f"{source_id}_normal_gold_vs_refusal_anchor_normal",
            "source_id": source_id,
            "candidate_type": "normal_gold_vs_refusal_anchor",
            "setting": "normal",
            "question": "",
            "gold": gold,
            "model_answer": NEUTRAL_ABSTAIN,
            "normal_answer": "",
            "text_only_answer": "",
            "wrong_image_answer": "",
            "blank_image_answer": "",
            "normal_f1": 1.0,
            "text_only_f1": 0.0,
            "wrong_image_f1": 0.0,
            "blank_image_f1": 0.0,
            "control_f1": 0.0,
            "case_gap": 0.0,
            "is_direct_answer": False,
            "is_short_prior_answer": False,
            "is_caution": True,
            "source_file": "data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl",
            "heldout_excluded": True,
            "rejected_from_actual_prediction": False,
            "trigger_reason": "normal gold vs neutral refusal anchor",
        })
    return candidates, removed


def build_stats(candidates: list[dict[str, Any]], diagnostic: dict[str, Any], removed: int, anchor_removed: int, heldout_keys: set[str]) -> dict[str, Any]:
    counts = Counter(str(row["candidate_type"]) for row in candidates)
    leaked = [row["source_id"] for row in candidates if identity_keys(row["source_id"]) & heldout_keys]
    return {
        "total_candidates": len(candidates),
        "candidate_type_counts": dict(counts),
        "candidate_type_ratios": {key: value / len(candidates) if candidates else 0.0 for key, value in sorted(counts.items())},
        "actual_prediction_candidate_count": sum(bool(row.get("rejected_from_actual_prediction")) for row in candidates),
        "heldout_removed_from_mining": removed,
        "heldout_removed_from_normal_anchors": anchor_removed,
        "heldout_leakage_count": len(leaked),
        "heldout_leakage_ids": leaked,
        "diagnostic_inputs": diagnostic,
        "warnings": ["Stage 8.5 held-out diagnostic rows are report-only and never used as pair sources."],
    }


def write_report(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preference-v8.2 Candidate Report",
        "",
        "Stage 8.5 held-out files were read only to preserve the error-attribution context. No held-out row is a training candidate.",
        "",
        f"- total_candidates: {stats['total_candidates']}",
        f"- heldout_removed_from_mining: {stats['heldout_removed_from_mining']}",
        f"- heldout_removed_from_normal_anchors: {stats['heldout_removed_from_normal_anchors']}",
        f"- heldout_leakage_count: {stats['heldout_leakage_count']}",
        f"- actual_prediction_candidate_count: {stats['actual_prediction_candidate_count']}",
        "",
        "## Candidate Types",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(stats["candidate_type_counts"].items()))
    lines += ["", "## Leakage Boundary", "- Stage 8.5 regression/fix/break records are held-out diagnostics only.", "- Candidate answers come from the non-heldout r3 mining pool or normal r3 training anchors."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect non-heldout Preference-v8.2 candidates.")
    parser.add_argument("--case_gap_regression", default="outputs/final_report/stage8_5_case_gap_regression.csv")
    parser.add_argument("--control_by_case", default="outputs/final_report/stage8_5_control_behavior_by_case.csv")
    parser.add_argument("--dpo_break_cases", default="outputs/final_report/stage8_5_dpo_break_cases.csv")
    parser.add_argument("--dpo_fix_cases", default="outputs/final_report/stage8_5_dpo_fix_cases.csv")
    parser.add_argument("--hard_negatives", default="data/mining/v8_1/hard_negatives_v8_1.jsonl")
    parser.add_argument("--sft_r3_data", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output_dir", default="data/mining/v8_2")
    parser.add_argument("--report", default="outputs/data_audit/preference_v8_2_candidate_report.md")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostic = read_diagnostic_counts([Path(args.case_gap_regression), Path(args.control_by_case), Path(args.dpo_break_cases), Path(args.dpo_fix_cases)])
    _, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    hard_rows = read_jsonl(Path(args.hard_negatives))
    sft_rows = read_jsonl(Path(args.sft_r3_data))
    if args.dry_run:
        hard_rows = hard_rows[: min(180, len(hard_rows))]
        sft_rows = sft_rows[: min(100, len(sft_rows))]
    cases, removed = consolidate_hard_rows(hard_rows, heldout_keys)
    candidates = collect_mined_candidates(cases)
    anchors, anchor_removed = collect_refusal_anchors(sft_rows, heldout_keys)
    candidates.extend(anchors)
    random.Random(args.seed).shuffle(candidates)
    stats = build_stats(candidates, diagnostic, removed, anchor_removed, heldout_keys)
    if stats["heldout_leakage_count"]:
        raise SystemExit("held-out leakage remains in Preference-v8.2 candidates; construction is blocked")
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "preference_v8_2_candidate_cases.jsonl", candidates)
    write_json(output_dir / "preference_v8_2_candidate_stats.json", stats)
    write_report(stats, Path(args.report))
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
