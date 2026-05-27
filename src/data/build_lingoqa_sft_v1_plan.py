"""Build a planning CSV for LingoQA SFT-v1 data curation.

This script does not build a training JSONL. It audits the existing SFT-v0
assets and the current candidate pool, then writes a plan that separates plain
SFT candidates from samples that need rewriting or a preference objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_MIX: dict[str, dict[str, int]] = {
    "positive_visual_grounding": {"min": 30, "max": 40},
    "spatial_reasoning_correction": {"min": 20, "max": 30},
    "anti_hallucination_counterfactual": {"min": 10, "max": 15},
    "high_quality_fix_rewritten": {"min": 7, "max": 15},
    "synthetic_drivemind_safety_tool_risk": {"min": 0, "max": 10},
}

CSV_FIELDNAMES = [
    "id",
    "eval_case_id",
    "source_status",
    "candidate_type",
    "capability",
    "sft_v1_role",
    "recommended_action",
    "ordinary_sft_eligible",
    "preferred_objective",
    "manual_rewrite_required",
    "priority_bucket",
    "priority_score",
    "visual_dependency_gap",
    "normal_f1",
    "control_max_f1",
    "text_only_f1",
    "wrong_image_f1",
    "blank_image_f1",
    "question",
    "gold_answer",
    "normal_prediction",
    "human_note",
    "suggested_rewrite",
    "plan_reason",
    "image",
    "image_paths",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a LingoQA SFT-v1 candidate plan.")
    parser.add_argument(
        "--candidate_jsonl",
        type=Path,
        default=Path("data/processed/lingoqa_sft_candidates_needs_review.jsonl"),
    )
    parser.add_argument(
        "--review_csv",
        type=Path,
        default=Path("outputs/cases/lingoqa_sft_candidate_review_agent_checked.csv"),
    )
    parser.add_argument(
        "--keep_jsonl",
        type=Path,
        default=Path("data/processed/lingoqa_sft_v0_keep_25.jsonl"),
    )
    parser.add_argument(
        "--fix_csv",
        type=Path,
        default=Path("outputs/cases/lingoqa_sft_v0_need_fix_7.csv"),
    )
    parser.add_argument(
        "--drop_csv",
        type=Path,
        default=Path("outputs/cases/lingoqa_sft_v0_drop_9.csv"),
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("outputs/cases/lingoqa_sft_v1_candidate_plan.csv"),
    )
    parser.add_argument(
        "--summary_output",
        type=Path,
        default=Path("outputs/eval_results/lingoqa_sft_v1_candidate_plan_summary.json"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))


def nested_get(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def answer_text(candidate: dict[str, Any]) -> str:
    answer = candidate.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("answer") or answer.get("reason") or "")
    return str(answer or "")


def eval_case_id(candidate_id: str) -> str:
    parts = candidate_id.split("_")
    if len(parts) >= 3 and parts[0] == "lingoqa" and parts[1] == "eval":
        return "_".join(parts[:3])
    return candidate_id


def image_paths_json(candidate: dict[str, Any]) -> str:
    image_paths = nested_get(candidate, ["meta", "external", "image_paths"], [])
    if not image_paths:
        return ""
    return json.dumps(image_paths, ensure_ascii=False)


def source_status(
    candidate_id: str,
    review_by_id: dict[str, dict[str, str]],
    keep_ids: set[str],
    fix_by_id: dict[str, dict[str, str]],
    drop_by_id: dict[str, dict[str, str]],
) -> str:
    if candidate_id in keep_ids:
        return "v0_keep"
    if candidate_id in fix_by_id:
        return "v0_fix"
    if candidate_id in drop_by_id:
        return "v0_drop"
    review_row = review_by_id.get(candidate_id)
    if review_row:
        return "reviewed_" + str(review_row.get("human_action", "unknown") or "unknown")
    return "unreviewed"


def suggested_rewrite(review_row: dict[str, str] | None, fix_by_id: dict[str, dict[str, str]], candidate_id: str) -> str:
    if candidate_id in fix_by_id:
        return fix_by_id[candidate_id].get("suggested_fix", "")
    if review_row:
        note = review_row.get("human_note", "")
        marker = "Rewrite as:"
        if marker in note:
            return note.split(marker, 1)[1].strip()
    return ""


def recommend(candidate: dict[str, Any], status: str) -> dict[str, str]:
    ctype = str(nested_get(candidate, ["review", "candidate_type"], ""))
    capability = str(nested_get(candidate, ["meta", "capability"], ""))
    gap = to_float(nested_get(candidate, ["review", "visual_dependency_gap"], nested_get(candidate, ["meta", "visual_dependency_gap"], 0.0)))
    normal_f1 = to_float(nested_get(candidate, ["review", "normal_f1"], nested_get(candidate, ["meta", "normal_f1"], 0.0)))
    control_max = to_float(nested_get(candidate, ["review", "control_max_f1"], nested_get(candidate, ["meta", "control_max_f1"], 0.0)))

    if status == "v0_drop":
        return {
            "sft_v1_role": "exclude",
            "recommended_action": "exclude_from_sft_v1",
            "ordinary_sft_eligible": "no",
            "preferred_objective": "none",
            "manual_rewrite_required": "no",
            "priority_bucket": "exclude",
            "plan_reason": "Previously reviewed as drop; keep only as an audit example unless a human reopens it.",
        }

    if status == "v0_fix":
        objective = "normal_vqa_sft_after_rewrite"
        if ctype == "wrong_image_confound":
            objective = "rewrite_then_preference_or_capped_rejection_sft"
        return {
            "sft_v1_role": "high_quality_fix_rewritten",
            "recommended_action": "rewrite_then_second_review",
            "ordinary_sft_eligible": "after_rewrite",
            "preferred_objective": objective,
            "manual_rewrite_required": "yes",
            "priority_bucket": "high",
            "plan_reason": "Existing fix row: promote only after rewriting the answer/reason with explicit visual evidence.",
        }

    if ctype == "positive_grounding":
        priority = "high" if gap >= 0.2 and normal_f1 >= 0.2 else "medium"
        action = "include_after_visual_verification" if status == "v0_keep" else "manual_review_for_positive_sft"
        eligible = "yes" if status == "v0_keep" else "after_review"
        return {
            "sft_v1_role": "positive_visual_grounding",
            "recommended_action": action,
            "ordinary_sft_eligible": eligible,
            "preferred_objective": "normal_vqa_sft",
            "manual_rewrite_required": "verify_answer_quality",
            "priority_bucket": priority,
            "plan_reason": "Positive per-case visual gap; use to increase normal-image grounding coverage after visual evidence check.",
        }

    if ctype == "hard_negative_spatial_reasoning":
        eligible = "yes_cautiously" if status == "v0_keep" else "after_rewrite"
        rewrite = "review_answer_evidence" if status == "v0_keep" else "yes"
        priority = "high" if capability in {"spatial_localization", "reasoning_world_knowledge"} else "medium"
        return {
            "sft_v1_role": "spatial_reasoning_correction",
            "recommended_action": "include_as_spatial_correction" if status == "v0_keep" else "rewrite_spatial_evidence_then_review",
            "ordinary_sft_eligible": eligible,
            "preferred_objective": "normal_vqa_sft_with_explicit_spatial_evidence",
            "manual_rewrite_required": rewrite,
            "priority_bucket": priority,
            "plan_reason": "Targets negative spatial/reasoning gaps; answer should state the visible objects, relative positions, and driving implication.",
        }

    if ctype == "wrong_image_confound":
        priority = "medium" if status == "v0_keep" else "low"
        return {
            "sft_v1_role": "anti_hallucination_counterfactual",
            "recommended_action": "cap_in_plain_sft_or_convert_to_preference",
            "ordinary_sft_eligible": "capped_only",
            "preferred_objective": "contrastive_preference_or_rejection_uncertainty",
            "manual_rewrite_required": "convert_to_contrastive_pair_preferred",
            "priority_bucket": priority,
            "plan_reason": "Wrong-image control is competitive; plain SFT has no penalty for the wrong image also producing the gold answer.",
        }

    if ctype in {"language_prior_confound", "blank_image_confound"}:
        return {
            "sft_v1_role": "language_prior_or_blank_rejection",
            "recommended_action": "do_not_use_as_plain_sft",
            "ordinary_sft_eligible": "no",
            "preferred_objective": "rejection_uncertainty_or_preference",
            "manual_rewrite_required": "yes_if_reopened",
            "priority_bucket": "low",
            "plan_reason": "Control answerability suggests language or blank-image priors; not suitable for ordinary positive SFT.",
        }

    if ctype == "all_settings_failed":
        priority = "medium" if capability in {"spatial_localization", "reasoning_world_knowledge"} else "low"
        return {
            "sft_v1_role": "manual_hard_case_review",
            "recommended_action": "inspect_for_rewrite_or_drop",
            "ordinary_sft_eligible": "after_rewrite",
            "preferred_objective": "normal_vqa_sft_after_rewrite_if_visual_evidence_is_clear",
            "manual_rewrite_required": "yes",
            "priority_bucket": priority,
            "plan_reason": "All settings failed; useful only if a human can verify clear visual evidence and repair the target answer.",
        }

    return {
        "sft_v1_role": "manual_review_mixed",
        "recommended_action": "manual_review",
        "ordinary_sft_eligible": "after_review",
        "preferred_objective": "undecided",
        "manual_rewrite_required": "review",
        "priority_bucket": "low",
        "plan_reason": "Mixed candidate type; requires manual decision before training use.",
    }


def build_plan_rows(
    candidates: list[dict[str, Any]],
    review_by_id: dict[str, dict[str, str]],
    keep_ids: set[str],
    fix_by_id: dict[str, dict[str, str]],
    drop_by_id: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("id", ""))
        review_row = review_by_id.get(candidate_id)
        status = source_status(candidate_id, review_by_id, keep_ids, fix_by_id, drop_by_id)
        rec = recommend(candidate, status)
        review = candidate.get("review", {}) if isinstance(candidate.get("review"), dict) else {}
        meta = candidate.get("meta", {}) if isinstance(candidate.get("meta"), dict) else {}
        human_note = ""
        if review_row:
            human_note = review_row.get("human_note", "")
        elif isinstance(review, dict):
            human_note = str(review.get("human_note", ""))

        row = {
            "id": candidate_id,
            "eval_case_id": eval_case_id(candidate_id),
            "source_status": status,
            "candidate_type": str(review.get("candidate_type", meta.get("curation_type", ""))),
            "capability": str(meta.get("capability", nested_get(candidate, ["answer", "subcategory"], ""))),
            "sft_v1_role": rec["sft_v1_role"],
            "recommended_action": rec["recommended_action"],
            "ordinary_sft_eligible": rec["ordinary_sft_eligible"],
            "preferred_objective": rec["preferred_objective"],
            "manual_rewrite_required": rec["manual_rewrite_required"],
            "priority_bucket": rec["priority_bucket"],
            "priority_score": str(review.get("priority_score", "")),
            "visual_dependency_gap": str(review.get("visual_dependency_gap", meta.get("visual_dependency_gap", ""))),
            "normal_f1": str(review.get("normal_f1", meta.get("normal_f1", ""))),
            "control_max_f1": str(review.get("control_max_f1", meta.get("control_max_f1", ""))),
            "text_only_f1": str(review.get("text_only_f1", "")),
            "wrong_image_f1": str(review.get("wrong_image_f1", "")),
            "blank_image_f1": str(review.get("blank_image_f1", "")),
            "question": str(candidate.get("instruction", "")),
            "gold_answer": answer_text(candidate),
            "normal_prediction": str(review.get("normal_prediction", "")).replace("\n", " "),
            "human_note": human_note,
            "suggested_rewrite": suggested_rewrite(review_row, fix_by_id, candidate_id),
            "plan_reason": rec["plan_reason"],
            "image": str(candidate.get("image", "")),
            "image_paths": image_paths_json(candidate),
        }
        rows.append(row)

    bucket_order = {"high": 0, "medium": 1, "low": 2, "exclude": 3}
    rows.sort(
        key=lambda row: (
            bucket_order.get(row["priority_bucket"], 9),
            -to_float(row.get("priority_score", 0.0)),
            row["candidate_type"],
            row["id"],
        )
    )
    return rows


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(counter.most_common())


def crosstab(rows: list[dict[str, str]], row_key: str, col_key: str) -> dict[str, dict[str, int]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        table[str(row.get(row_key, ""))][str(row.get(col_key, ""))] += 1
    return {key: counter_dict(value) for key, value in sorted(table.items())}


def target_gap_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    role_counts = Counter(row["sft_v1_role"] for row in rows if row["source_status"] != "v0_drop")
    ready_counts = Counter(
        row["sft_v1_role"]
        for row in rows
        if row["source_status"] == "v0_keep" and row["ordinary_sft_eligible"] in {"yes", "yes_cautiously", "capped_only"}
    )
    fix_counts = Counter(row["sft_v1_role"] for row in rows if row["source_status"] == "v0_fix")

    summary: dict[str, Any] = {}
    for role, target in TARGET_MIX.items():
        available = role_counts.get(role, 0)
        ready = ready_counts.get(role, 0)
        fix_available = fix_counts.get(role, 0)
        min_target = target["min"]
        max_target = target["max"]
        summary[role] = {
            "target_min": min_target,
            "target_max": max_target,
            "ready_now": ready,
            "available_in_current_pool_after_review_or_rewrite": available,
            "fix_rows_available": fix_available,
            "shortage_vs_min_if_using_current_pool": max(0, min_target - available),
            "excess_vs_max_ready_now": max(0, ready - max_target),
        }
    return summary


def summarize(
    candidates: list[dict[str, Any]],
    review_rows: list[dict[str, str]],
    keep_rows: list[dict[str, Any]],
    fix_rows: list[dict[str, str]],
    drop_rows: list[dict[str, str]],
    plan_rows: list[dict[str, str]],
) -> dict[str, Any]:
    candidate_type_counter = Counter(str(nested_get(row, ["review", "candidate_type"], "")) for row in candidates)
    candidate_cap_counter = Counter(str(nested_get(row, ["meta", "capability"], "")) for row in candidates)
    keep_role_counter = Counter(str(nested_get(row, ["meta", "sft_v0_role"], "")) for row in keep_rows)
    keep_cap_counter = Counter(str(nested_get(row, ["meta", "capability"], "")) for row in keep_rows)

    return {
        "inputs": {
            "candidate_pool_count": len(candidates),
            "reviewed_count": len(review_rows),
            "v0_keep_count": len(keep_rows),
            "v0_fix_count": len(fix_rows),
            "v0_drop_count": len(drop_rows),
        },
        "candidate_pool_by_type": counter_dict(candidate_type_counter),
        "candidate_pool_by_capability": counter_dict(candidate_cap_counter),
        "v0_keep_by_role": counter_dict(keep_role_counter),
        "v0_keep_by_capability": counter_dict(keep_cap_counter),
        "plan_by_source_status": counter_dict(Counter(row["source_status"] for row in plan_rows)),
        "plan_by_sft_v1_role": counter_dict(Counter(row["sft_v1_role"] for row in plan_rows)),
        "plan_by_recommended_action": counter_dict(Counter(row["recommended_action"] for row in plan_rows)),
        "plan_by_preferred_objective": counter_dict(Counter(row["preferred_objective"] for row in plan_rows)),
        "role_by_capability": crosstab(plan_rows, "sft_v1_role", "capability"),
        "action_by_candidate_type": crosstab(plan_rows, "candidate_type", "recommended_action"),
        "target_mix": TARGET_MIX,
        "target_gap_summary": target_gap_summary(plan_rows),
        "warnings": [
            "This script creates a curation plan, not a final training set.",
            "Current candidates come from the 100-sample control workflow; do not use them to claim held-out benchmark gains unless evaluation is moved to a clean holdout.",
            "wrong_image_confound rows should be capped in ordinary SFT or converted to contrastive preference/rejection data.",
            "positive_visual_grounding remains below the target minimum even if all current positive candidates are accepted.",
        ],
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    require_inputs([args.candidate_jsonl, args.review_csv, args.keep_jsonl, args.fix_csv, args.drop_csv])

    candidates = load_jsonl(args.candidate_jsonl)
    review_rows = load_csv(args.review_csv)
    keep_rows = load_jsonl(args.keep_jsonl)
    fix_rows = load_csv(args.fix_csv)
    drop_rows = load_csv(args.drop_csv)

    review_by_id = {row["id"]: row for row in review_rows if row.get("id")}
    keep_ids = {str(row.get("id", "")) for row in keep_rows}
    fix_by_id = {row["id"]: row for row in fix_rows if row.get("id")}
    drop_by_id = {row["id"]: row for row in drop_rows if row.get("id")}

    plan_rows = build_plan_rows(candidates, review_by_id, keep_ids, fix_by_id, drop_by_id)
    summary = summarize(candidates, review_rows, keep_rows, fix_rows, drop_rows, plan_rows)

    write_csv(args.output_csv, plan_rows)
    write_json(args.summary_output, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote plan CSV to {args.output_csv}")
    print(f"wrote summary JSON to {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
