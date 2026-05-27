"""Build DriveLM visual-dependency preference pairs from visual-control failures."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.output_parser import parse_model_output
from src.eval.refusal_detection import is_refusal_prediction
from src.eval.run_all_eval import token_f1


CONTROL_MODES = ("text_only", "wrong_image", "blank_image")
FAILURE_PRIORITY = (
    "blank_beats_normal",
    "text_beats_normal",
    "wrong_image_beats_normal",
    "wrong_image_invariant_low",
    "normal_low",
)


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


def parse_answer_obj(value: Any) -> dict[str, str]:
    parsed = parse_model_output(value)
    if parsed.get("ok") and isinstance(parsed.get("data"), dict):
        data = parsed["data"]
        text = str(data.get("answer") or data.get("reason") or "").strip()
        reason = str(data.get("reason") or text).strip()
        return {"task": str(data.get("task") or "external_vqa"), "answer": text, "reason": reason}
    text = str(value or "").strip()
    return {"task": "external_vqa", "answer": text, "reason": text}


def gold_answer(row: dict[str, Any]) -> dict[str, str]:
    answer = row.get("answer")
    if not isinstance(answer, dict):
        answer = row.get("gold") if isinstance(row.get("gold"), dict) else {}
    text = str(answer.get("answer") or answer.get("reason") or "").strip()
    reason = str(answer.get("reason") or text).strip()
    return {
        "task": str(answer.get("task") or "external_vqa"),
        "answer": text,
        "reason": reason,
    }


def pred_answer(row: dict[str, Any] | None) -> dict[str, str]:
    if not row:
        return {"task": "external_vqa", "answer": "", "reason": ""}
    return parse_answer_obj(row.get("prediction"))


def answer_text(answer: dict[str, Any]) -> str:
    return str(answer.get("answer") or answer.get("reason") or "")


def refusal_for(mode: str) -> dict[str, str]:
    if mode == "text_only":
        reason = "No image frames are available, so this visual driving question cannot be answered reliably."
    elif mode == "blank_image":
        reason = "The provided image frames are blank placeholders and contain no driving-scene evidence."
    elif mode == "wrong_image":
        reason = "The provided image frames do not match the question scene, so a scene-specific answer would be unreliable."
    else:
        reason = "The normal image input contains visual evidence, so refusing would discard usable information."
    return {
        "task": "external_vqa",
        "answer": "I cannot determine this from the provided visual input.",
        "reason": reason,
    }


def deep_copy_json(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def prompt_sample(row: dict[str, Any], mode: str) -> dict[str, Any]:
    sample = deep_copy_json(row)
    sample.pop("answer", None)
    sample.setdefault("meta", {})
    sample["meta"]["train_input_mode"] = mode
    return sample


def capability(row: dict[str, Any]) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    return str(meta.get("capability") or external.get("capability") or answer.get("subcategory") or "unknown")


def category(row: dict[str, Any]) -> str:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else {}
    external = row.get("meta", {}).get("external", {}) if isinstance(row.get("meta"), dict) else {}
    return str(answer.get("category") or external.get("category") or "unknown")


def case_score(case: dict[str, Any], mode: str) -> float:
    return float(case.get(f"{mode}_f1", 0.0) or 0.0)


def classify_case(case: dict[str, Any], tol: float, low_f1: float) -> str:
    normal = case_score(case, "normal")
    scores = {mode: case_score(case, mode) for mode in CONTROL_MODES}
    best_mode, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > normal + tol:
        if best_mode == "blank_image":
            return "blank_beats_normal"
        if best_mode == "text_only":
            return "text_beats_normal"
        return "wrong_image_beats_normal"
    if abs(scores["wrong_image"] - normal) <= tol and normal <= low_f1:
        return "wrong_image_invariant_low"
    if normal <= low_f1:
        return "normal_low"
    return "mixed_or_usable"


def make_pair(
    *,
    pair_id: str,
    source_id: str,
    source_row: dict[str, Any],
    chosen: dict[str, str],
    rejected: dict[str, str],
    pair_type: str,
    mode: str,
    weight: float,
    sft_anchor_weight: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    merged_meta = {
        "source": "drivelm_visual_preference_v1",
        "capability": capability(source_row),
        "category": category(source_row),
        "weight": round(weight, 4),
        "sft_anchor_weight": round(sft_anchor_weight, 4),
    }
    merged_meta.update(meta)
    return {
        "id": pair_id,
        "source_id": source_id,
        "pair_type": pair_type,
        "train_input_mode": mode,
        "weight": round(weight, 4),
        "sft_anchor_weight": round(sft_anchor_weight, 4),
        "prompt_sample": prompt_sample(source_row, mode),
        "chosen": chosen,
        "rejected": rejected,
        "meta": merged_meta,
    }


def add_normal_anchor_pair(
    pairs: list[dict[str, Any]],
    row: dict[str, Any],
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    gold = gold_answer(row)
    if not answer_text(gold):
        counters["normal_anchor_skipped_empty_gold"] += 1
        return
    source_id = str(row.get("id"))
    pairs.append(
        make_pair(
            pair_id=f"{source_id}__normal_anchor_gold_over_refusal",
            source_id=source_id,
            source_row=row,
            chosen=gold,
            rejected=refusal_for("normal"),
            pair_type="normal_anchor_gold_over_refusal",
            mode="normal",
            weight=args.normal_anchor_weight,
            sft_anchor_weight=args.normal_anchor_sft_weight,
            meta={"failure_type": "normal_anchor", "rejected_source": "template_refusal"},
        )
    )
    counters["normal_anchor_added"] += 1


def add_normal_repair_pair(
    pairs: list[dict[str, Any]],
    row: dict[str, Any],
    normal_pred_row: dict[str, Any] | None,
    case: dict[str, Any],
    failure_type: str,
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    if normal_pred_row is None:
        counters["normal_repair_skipped_missing_prediction"] += 1
        return
    gold = gold_answer(row)
    rejected = pred_answer(normal_pred_row)
    if not answer_text(gold) or not answer_text(rejected):
        counters["normal_repair_skipped_empty_answer"] += 1
        return
    if token_f1(answer_text(gold), answer_text(rejected)) >= args.max_normal_repair_rejected_f1:
        counters["normal_repair_skipped_rejected_too_good"] += 1
        return
    source_id = str(row.get("id"))
    pairs.append(
        make_pair(
            pair_id=f"{source_id}__normal_repair_gold_over_bad_prediction",
            source_id=source_id,
            source_row=row,
            chosen=gold,
            rejected=rejected,
            pair_type="normal_repair_gold_over_bad_prediction",
            mode="normal",
            weight=args.normal_repair_weight,
            sft_anchor_weight=args.normal_repair_sft_weight,
            meta={
                "failure_type": failure_type,
                "normal_f1": round(case_score(case, "normal"), 4),
                "control_max_f1": round(float(case.get("control_max_f1", 0.0) or 0.0), 4),
                "visual_dependency_gap": round(float(case.get("visual_dependency_gap", 0.0) or 0.0), 4),
                "rejected_source": "normal_prediction",
            },
        )
    )
    counters["normal_repair_added"] += 1


def add_control_refusal_pair(
    pairs: list[dict[str, Any]],
    mode: str,
    source_row: dict[str, Any] | None,
    pred_row: dict[str, Any] | None,
    gold: dict[str, str],
    case: dict[str, Any],
    failure_type: str,
    args: argparse.Namespace,
    counters: Counter[str],
) -> None:
    if source_row is None or pred_row is None:
        counters[f"{mode}_control_skipped_missing_source"] += 1
        return
    if is_refusal_prediction(pred_row.get("prediction")):
        counters[f"{mode}_control_skipped_already_refusal"] += 1
        return
    rejected = pred_answer(pred_row)
    if not answer_text(rejected):
        counters[f"{mode}_control_skipped_empty_prediction"] += 1
        return
    if args.min_control_prediction_f1 > 0:
        overlap = token_f1(answer_text(rejected), answer_text(gold))
        if overlap < args.min_control_prediction_f1:
            counters[f"{mode}_control_skipped_low_overlap"] += 1
            return
    source_id = str(source_row.get("id"))
    pairs.append(
        make_pair(
            pair_id=f"{source_id}__{mode}_refusal_over_control_prediction",
            source_id=source_id,
            source_row=source_row,
            chosen=refusal_for(mode),
            rejected=rejected,
            pair_type="control_refusal_over_prediction",
            mode=mode,
            weight=getattr(args, f"{mode}_weight"),
            sft_anchor_weight=0.0,
            meta={
                "failure_type": failure_type,
                "control_mode": mode,
                "normal_f1": round(case_score(case, "normal"), 4),
                "control_f1": round(case_score(case, mode), 4),
                "visual_dependency_gap": round(float(case.get("visual_dependency_gap", 0.0) or 0.0), 4),
                "rejected_source": f"{mode}_prediction",
            },
        )
    )
    counters[f"{mode}_control_added"] += 1


def build_pairs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normal_rows = read_jsonl(Path(args.normal_data))
    wrong_rows = read_jsonl(Path(args.wrong_data)) if args.wrong_data else []
    blank_rows = read_jsonl(Path(args.blank_data)) if args.blank_data else []
    cases = read_jsonl(Path(args.cases))
    normal_preds = index_by_id(read_jsonl(Path(args.normal_predictions)))
    text_preds = index_by_id(read_jsonl(Path(args.text_only_predictions)))
    wrong_preds = index_by_id(read_jsonl(Path(args.wrong_image_predictions)))
    blank_preds = index_by_id(read_jsonl(Path(args.blank_image_predictions)))

    normal_by_id = index_by_id(normal_rows)
    wrong_by_id = index_by_id(wrong_rows)
    blank_by_id = index_by_id(blank_rows)
    cases_by_id = index_by_id(cases)
    counters: Counter[str] = Counter()
    by_failure: Counter[str] = Counter()
    by_capability: Counter[str] = Counter()
    by_pair_type: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    pairs: list[dict[str, Any]] = []

    anchor_rows = list(normal_rows)
    random.Random(args.seed).shuffle(anchor_rows)
    if args.max_normal_anchor_pairs > 0:
        anchor_rows = anchor_rows[: args.max_normal_anchor_pairs]
    for row in anchor_rows:
        add_normal_anchor_pair(pairs, row, args, counters)

    serious_cases = []
    for case in cases:
        failure_type = classify_case(case, args.tol, args.low_f1)
        if failure_type in FAILURE_PRIORITY:
            serious_cases.append((failure_type, case))
    serious_cases.sort(key=lambda item: (FAILURE_PRIORITY.index(item[0]), float(item[1].get("visual_dependency_gap", 0.0) or 0.0)))
    if args.max_failure_cases > 0:
        serious_cases = serious_cases[: args.max_failure_cases]

    for failure_type, case in serious_cases:
        sample_id = str(case.get("id") or "")
        row = normal_by_id.get(sample_id)
        if row is None:
            counters["failure_skipped_missing_normal_row"] += 1
            continue
        by_failure[failure_type] += 1
        gold = gold_answer(row)
        if case_score(case, "normal") <= args.normal_repair_max_f1:
            add_normal_repair_pair(pairs, row, normal_preds.get(sample_id), case, failure_type, args, counters)

        if failure_type in {"text_beats_normal", "blank_beats_normal", "wrong_image_beats_normal", "wrong_image_invariant_low"}:
            add_control_refusal_pair(
                pairs,
                "text_only",
                row,
                text_preds.get(sample_id),
                gold,
                case,
                failure_type,
                args,
                counters,
            )
            add_control_refusal_pair(
                pairs,
                "wrong_image",
                wrong_by_id.get(sample_id),
                wrong_preds.get(sample_id),
                gold,
                case,
                failure_type,
                args,
                counters,
            )
            add_control_refusal_pair(
                pairs,
                "blank_image",
                blank_by_id.get(sample_id),
                blank_preds.get(sample_id),
                gold,
                case,
                failure_type,
                args,
                counters,
            )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        if pair["id"] in seen:
            counters["deduped_duplicate_pair_id"] += 1
            continue
        seen.add(pair["id"])
        deduped.append(pair)
        by_capability[str(pair.get("meta", {}).get("capability", "unknown"))] += 1
        by_pair_type[str(pair.get("pair_type", "unknown"))] += 1
        by_mode[str(pair.get("train_input_mode", "unknown"))] += 1
    random.Random(args.seed).shuffle(deduped)

    summary = {
        "pairs": len(deduped),
        "normal_rows": len(normal_rows),
        "case_rows": len(cases),
        "by_pair_type": dict(by_pair_type.most_common()),
        "by_train_input_mode": dict(by_mode.most_common()),
        "by_capability": dict(by_capability.most_common()),
        "selected_failure_cases": dict(by_failure.most_common()),
        "counters": dict(counters.most_common()),
        "inputs": {
            "normal_data": args.normal_data,
            "wrong_data": args.wrong_data,
            "blank_data": args.blank_data,
            "cases": args.cases,
            "normal_predictions": args.normal_predictions,
            "text_only_predictions": args.text_only_predictions,
            "wrong_image_predictions": args.wrong_image_predictions,
            "blank_image_predictions": args.blank_image_predictions,
        },
    }
    return deduped, summary


def write_markdown(path: Path, summary: dict[str, Any], output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def table(items: dict[str, Any]) -> list[str]:
        lines = ["| name | count |", "|---|---:|"]
        for key, value in items.items():
            lines.append(f"| {key} | {value} |")
        return lines

    lines = [
        "# DriveLM Visual Preference V1 Audit",
        "",
        f"Output: `{output}`",
        f"Pairs: `{summary['pairs']}`",
        "",
        "## Pair Types",
        "",
    ]
    lines.extend(table(summary["by_pair_type"]))
    lines.extend(["", "## Train Input Modes", ""])
    lines.extend(table(summary["by_train_input_mode"]))
    lines.extend(["", "## Capabilities", ""])
    lines.extend(table(summary["by_capability"]))
    lines.extend(["", "## Selected Failure Cases", ""])
    lines.extend(table(summary["selected_failure_cases"]))
    lines.extend(["", "## Counters", ""])
    lines.extend(table(summary["counters"]))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Normal-anchor pairs preserve the DriveLM answer style and reduce refusal drift.",
            "- Normal-repair pairs teach the model to prefer gold answers over its own bad normal-image predictions.",
            "- Control-refusal pairs teach text-only, wrong-image, and blank-image prompts to refuse scene-specific answers.",
            "- If this file was built from dev predictions, use it as a diagnostic smoke artifact only; final DPO data should be rebuilt from train-scene predictions to avoid evaluation leakage.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM visual-dependency preference pairs.")
    parser.add_argument("--normal_data", default="data/processed/drivelm_dev_scene.jsonl")
    parser.add_argument("--wrong_data", default="data/processed/drivelm_dev_scene_wrong_image.jsonl")
    parser.add_argument("--blank_data", default="data/processed/drivelm_dev_scene_blank_image.jsonl")
    parser.add_argument("--cases", default="outputs/cases/drivelm_dev_scene_sft_1200_visual_control_cases.jsonl")
    parser.add_argument("--normal_predictions", default="outputs/eval_results/drivelm_dev_scene_sft_1200_normal_predictions.jsonl")
    parser.add_argument("--text_only_predictions", default="outputs/eval_results/drivelm_dev_scene_sft_1200_text_only_predictions.jsonl")
    parser.add_argument("--wrong_image_predictions", default="outputs/eval_results/drivelm_dev_scene_sft_1200_wrong_predictions.jsonl")
    parser.add_argument("--blank_image_predictions", default="outputs/eval_results/drivelm_dev_scene_sft_1200_blank_predictions.jsonl")
    parser.add_argument("--output", default="data/processed/drivelm_visual_pref_v1_dev_diagnostic.jsonl")
    parser.add_argument("--summary", default="outputs/eval_results/drivelm_visual_pref_v1_dev_diagnostic_summary.json")
    parser.add_argument("--markdown", default="docs/drivelm_visual_pref_v1_dev_diagnostic.md")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--low_f1", type=float, default=0.25)
    parser.add_argument("--max_normal_anchor_pairs", type=int, default=300)
    parser.add_argument("--max_failure_cases", type=int, default=240)
    parser.add_argument("--normal_repair_max_f1", type=float, default=0.35)
    parser.add_argument("--max_normal_repair_rejected_f1", type=float, default=0.65)
    parser.add_argument("--min_control_prediction_f1", type=float, default=0.0)
    parser.add_argument("--normal_anchor_weight", type=float, default=0.6)
    parser.add_argument("--normal_anchor_sft_weight", type=float, default=1.0)
    parser.add_argument("--normal_repair_weight", type=float, default=1.0)
    parser.add_argument("--normal_repair_sft_weight", type=float, default=1.2)
    parser.add_argument("--text_only_weight", type=float, default=0.35)
    parser.add_argument("--wrong_image_weight", type=float, default=0.55)
    parser.add_argument("--blank_image_weight", type=float, default=0.45)
    args = parser.parse_args()

    pairs, summary = build_pairs(args)
    write_jsonl(Path(args.output), pairs)
    write_json(Path(args.summary), summary)
    write_markdown(Path(args.markdown), summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.output}, {args.summary}, {args.markdown}")


if __name__ == "__main__":
    main()
