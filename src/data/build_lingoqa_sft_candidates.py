"""Build reviewable SFT candidate data from LingoQA visual-control cases."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_all_eval import load_jsonl


TARGET_HARD_CAPABILITIES = {"spatial_localization", "reasoning_world_knowledge"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def index_dataset(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def candidate_type(case: dict[str, Any], pass_threshold: float, positive_margin: float) -> str:
    capability = str(case.get("capability", ""))
    normal_f1 = float(case.get("normal_f1", 0.0))
    gap = float(case.get("visual_dependency_gap", 0.0))
    wrong_f1 = float(case.get("wrong_image_f1", 0.0))
    blank_f1 = float(case.get("blank_image_f1", 0.0))
    text_f1 = float(case.get("text_only_f1", 0.0))
    control_max = max(text_f1, wrong_f1, blank_f1)

    if gap >= positive_margin and normal_f1 >= pass_threshold:
        return "positive_grounding"
    if capability in TARGET_HARD_CAPABILITIES and gap < 0:
        return "hard_negative_spatial_reasoning"
    if wrong_f1 >= normal_f1 and wrong_f1 >= pass_threshold:
        return "wrong_image_confound"
    if text_f1 >= normal_f1 and text_f1 >= pass_threshold:
        return "language_prior_confound"
    if blank_f1 >= normal_f1 and blank_f1 >= pass_threshold:
        return "blank_image_confound"
    if normal_f1 < pass_threshold and control_max < pass_threshold:
        return "all_settings_failed"
    return "mixed_review"


def priority_score(case: dict[str, Any], ctype: str) -> float:
    gap = float(case.get("visual_dependency_gap", 0.0))
    normal_f1 = float(case.get("normal_f1", 0.0))
    control_max = float(case.get("control_max_f1", 0.0))
    weights = {
        "hard_negative_spatial_reasoning": 5.0,
        "wrong_image_confound": 4.0,
        "language_prior_confound": 3.5,
        "blank_image_confound": 3.0,
        "all_settings_failed": 2.5,
        "positive_grounding": 2.0,
        "mixed_review": 1.0,
    }
    if ctype == "positive_grounding":
        return weights[ctype] + gap + normal_f1
    return weights.get(ctype, 1.0) + abs(min(gap, 0.0)) + control_max - normal_f1


def build_candidate(case: dict[str, Any], sample: dict[str, Any], pass_threshold: float, positive_margin: float) -> dict[str, Any]:
    ctype = candidate_type(case, pass_threshold, positive_margin)
    answer = sample.get("answer") if isinstance(sample.get("answer"), dict) else case.get("gold", {})
    meta = dict(sample.get("meta", {})) if isinstance(sample.get("meta"), dict) else {}
    external = meta.get("external", {}) if isinstance(meta.get("external"), dict) else {}
    meta.update(
        {
            "source": "lingoqa_visual_control_candidate",
            "curation_type": ctype,
            "curation_status": "needs_human_review",
            "benchmark_source": "lingoqa",
            "capability": case.get("capability", external.get("capability", "")),
            "visual_dependency_gap": case.get("visual_dependency_gap", 0.0),
            "normal_f1": case.get("normal_f1", 0.0),
            "control_max_f1": case.get("control_max_f1", 0.0),
            "prompt_variant": "spatial",
            "frame_strategy": "first_middle_last",
            "max_images": 3,
        }
    )
    return {
        "id": f"{sample.get('id', case.get('id'))}_{ctype}",
        "image": sample.get("image", ""),
        "video": sample.get("video", ""),
        "vehicle_state": sample.get("vehicle_state", {}),
        "perception": sample.get("perception", {}),
        "instruction": sample.get("instruction", ""),
        "answer": answer,
        "meta": meta,
        "review": {
            "candidate_type": ctype,
            "priority_score": round(priority_score(case, ctype), 4),
            "normal_prediction": case.get("normal_prediction", ""),
            "normal_f1": case.get("normal_f1", 0.0),
            "text_only_f1": case.get("text_only_f1", 0.0),
            "wrong_image_f1": case.get("wrong_image_f1", 0.0),
            "blank_image_f1": case.get("blank_image_f1", 0.0),
            "visual_dependency_gap": case.get("visual_dependency_gap", 0.0),
            "human_action": "keep_fix_or_drop",
            "human_note": "",
        },
    }


def build_review_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in candidates:
        review = item.get("review", {})
        answer = item.get("answer", {})
        meta = item.get("meta", {})
        rows.append(
            {
                "id": item.get("id", ""),
                "candidate_type": review.get("candidate_type", ""),
                "priority_score": review.get("priority_score", ""),
                "capability": meta.get("capability", ""),
                "visual_dependency_gap": review.get("visual_dependency_gap", ""),
                "normal_f1": review.get("normal_f1", ""),
                "control_max_f1": meta.get("control_max_f1", ""),
                "question": item.get("instruction", ""),
                "gold_answer": answer.get("answer", "") if isinstance(answer, dict) else "",
                "normal_prediction": str(review.get("normal_prediction", "")).replace("\n", " "),
                "human_action": review.get("human_action", ""),
                "human_note": "",
            }
        )
    return rows


def summarize(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(item.get("review", {}).get("candidate_type", "") for item in candidates)
    by_capability = Counter(item.get("meta", {}).get("capability", "") for item in candidates)
    by_type_capability: dict[str, Counter[str]] = defaultdict(Counter)
    for item in candidates:
        ctype = item.get("review", {}).get("candidate_type", "")
        capability = item.get("meta", {}).get("capability", "")
        by_type_capability[ctype][capability] += 1
    return {
        "count": len(candidates),
        "by_candidate_type": dict(by_type.most_common()),
        "by_capability": dict(by_capability.most_common()),
        "candidate_type_by_capability": {
            ctype: dict(counter.most_common()) for ctype, counter in sorted(by_type_capability.items())
        },
    }


def write_report(path: Path, summary: dict[str, Any], review_rows: list[dict[str, Any]], max_examples: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LingoQA SFT Candidate Curation Report",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Review Policy",
        "",
        "- `positive_grounding`: keep as evidence-style SFT examples after checking the visual answer.",
        "- `hard_negative_spatial_reasoning`: highest priority for manual correction; these target the current main weakness.",
        "- `wrong_image_confound` / `language_prior_confound` / `blank_image_confound`: use to design contrastive prompts or reject unsupported visual claims.",
        "- `all_settings_failed`: inspect for ambiguous reference, low image quality, or genuinely difficult samples before training.",
        "",
        "## Top Review Examples",
        "",
        "| id | type | capability | gap | normal_f1 | question | gold_answer |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in review_rows[:max_examples]:
        lines.append(
            "| {id} | {candidate_type} | {capability} | {visual_dependency_gap} | {normal_f1} | {question} | {gold_answer} |".format(
                **{key: str(value).replace("\n", " ")[:160] for key, value in row.items()}
            )
        )
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "Manually review the CSV, fill `human_action` with `keep`, `fix`, or `drop`, then build a small SFT set from approved rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LingoQA SFT candidates from strict visual-control cases.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_jsonl", default="data/processed/lingoqa_sft_candidates_needs_review.jsonl")
    parser.add_argument("--review_csv", default="outputs/cases/lingoqa_sft_candidate_review.csv")
    parser.add_argument("--summary", default="outputs/eval_results/lingoqa_sft_candidate_summary.json")
    parser.add_argument("--report", default="docs/lingoqa_sft_candidate_curation.md")
    parser.add_argument("--pass_threshold", type=float, default=0.2)
    parser.add_argument("--positive_margin", type=float, default=0.05)
    args = parser.parse_args()

    dataset_by_id = index_dataset(Path(args.dataset))
    cases = load_jsonl(Path(args.cases))
    candidates = []
    for case in cases:
        sample = dataset_by_id.get(str(case.get("id")))
        if not sample:
            continue
        candidates.append(build_candidate(case, sample, args.pass_threshold, args.positive_margin))
    candidates.sort(key=lambda item: float(item.get("review", {}).get("priority_score", 0.0)), reverse=True)

    review_rows = build_review_rows(candidates)
    summary = summarize(candidates)
    write_jsonl(Path(args.output_jsonl), candidates)
    write_csv(Path(args.review_csv), review_rows)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(Path(args.report), summary, review_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote candidates to {args.output_jsonl}")
    print(f"wrote review CSV to {args.review_csv}")
    print(f"wrote curation report to {args.report}")


if __name__ == "__main__":
    main()
