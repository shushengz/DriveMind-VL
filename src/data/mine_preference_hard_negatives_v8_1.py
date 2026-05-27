"""Mine r3 hard negatives from saved non-held-out strict visual-control predictions."""
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

from src.data.exclude_heldout_ids import identity_keys, read_heldout_keys, read_jsonl, write_jsonl
from src.eval.rescore_answer_only import SETTINGS, align_settings, load_model_settings, score_prediction

SPATIAL_RE = re.compile(r"\b(?:left|right|front|back|lane|pedestrian|vehicle|cyclist|car)\b|traffic light", re.I)
CONFIDENT_RE = re.compile(
    r"^\s*(?:yes|no|true|false)\b|\b(?:turn|stop|slow|go|accelerate|brake|"
    r"one|two|three|four|red|green|yellow|left|right|front|behind|ahead)\b",
    re.I,
)


def read_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"missing mining pool IDs: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"mining ID file has no ids: {path}")
    return [str(sample_id) for sample_id in ids]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def answer_values(group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {setting: score_prediction(group[setting], "answer_only") for setting in SETTINGS}


def similar(a: str, b: str) -> bool:
    left = set(re.findall(r"\w+", a.lower()))
    right = set(re.findall(r"\w+", b.lower()))
    return bool(left or right) and len(left & right) / max(1, len(left | right)) >= 0.75


def hard_row(
    source_id: str,
    hard_type: str,
    setting: str,
    group: dict[str, dict[str, Any]],
    values: dict[str, dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    controls = [values[item]["f1"] for item in ("text_only", "wrong_image", "blank_image")]
    return {
        "id": f"{source_id}_{hard_type}",
        "source_id": source_id,
        "hard_type": hard_type,
        "setting": setting,
        "question": group["normal"].get("question", ""),
        "gold": group["normal"].get("gold", ""),
        "normal_answer": values["normal"]["score_text"],
        "text_only_answer": values["text_only"]["score_text"],
        "wrong_image_answer": values["wrong_image"]["score_text"],
        "blank_image_answer": values["blank_image"]["score_text"],
        "normal_f1": values["normal"]["f1"],
        "text_only_f1": values["text_only"]["f1"],
        "wrong_image_f1": values["wrong_image"]["f1"],
        "blank_image_f1": values["blank_image"]["f1"],
        "case_gap": values["normal"]["f1"] - max(controls),
        "trigger_reason": reason,
        "heldout_excluded": True,
        "rejected_from_r3_actual_prediction": True,
    }


def mine(
    aligned: list[dict[str, dict[str, Any]]],
    expected_ids: list[str],
    heldout_keys: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {str(group["normal"].get("id")): group for group in aligned}
    missing = [sample_id for sample_id in expected_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"predictions are missing mining IDs: {missing[:5]}")
    leaking = [sample_id for sample_id in expected_ids if identity_keys(sample_id, by_id[sample_id]["normal"]) & heldout_keys]
    if leaking:
        raise ValueError(f"held-out leakage detected in mining predictions: {leaking[:5]}")
    rows: list[dict[str, Any]] = []
    for source_id in expected_ids:
        group = by_id[source_id]
        values = answer_values(group)
        normal = values["normal"]
        text = values["text_only"]
        wrong = values["wrong_image"]
        blank = values["blank_image"]
        question = str(group["normal"].get("question", ""))
        if not text["refusal"] and (
            text["f1"] >= 0.2 or text["f1"] >= normal["f1"] or CONFIDENT_RE.search(text["score_text"])
        ):
            rows.append(hard_row(source_id, "text_prior_bias", "text_only", group, values, "non-refusal text-only answer with high overlap or confident content"))
        if not wrong["refusal"] and (
            wrong["f1"] >= normal["f1"] - 0.05 or wrong["f1"] >= 0.25 or similar(wrong["score_text"], normal["score_text"])
        ):
            rows.append(hard_row(source_id, "wrong_image_confound", "wrong_image", group, values, "wrong-image answer is competitive with normal or highly similar"))
        if not blank["refusal"] and (blank["f1"] >= 0.2 or CONFIDENT_RE.search(blank["score_text"])):
            rows.append(hard_row(source_id, "blank_prior_answer", "blank_image", group, values, "non-refusal blank-image answer with overlap or confident content"))
        control = {setting: values[setting] for setting in ("text_only", "wrong_image", "blank_image")}
        best_setting = max(control, key=lambda item: control[item]["f1"])
        if control[best_setting]["f1"] >= normal["f1"] or control[best_setting]["f1"] >= 0.30:
            rows.append(hard_row(source_id, "control_over_gold_overlap", best_setting, group, values, "a control answer matches or exceeds the normal answer score"))
        if normal["f1"] < 0.2 and str(group["normal"].get("gold", "")).strip():
            rows.append(hard_row(source_id, "normal_wrong_anchor", "normal", group, values, "normal answer score below 0.2"))
        if SPATIAL_RE.search(question) and normal["f1"] < 0.3:
            rows.append(hard_row(source_id, "spatial_relation_error", "normal", group, values, "spatial question has low normal answer score"))
    counts = Counter(row["hard_type"] for row in rows)
    stats = {
        "mining_id_count": len(expected_ids),
        "hard_negative_count": len(rows),
        "hard_type_counts": dict(counts),
        "hard_type_ratios": {key: count / len(rows) if rows else 0.0 for key, count in counts.items()},
        "heldout_leakage_count": 0,
        "uses_model_predictions": True,
        "answer_only_mining": True,
    }
    return rows, stats


def write_markdown(path: Path, stats: dict[str, Any]) -> None:
    lines = [
        "# Hard Negatives v8.1 Report",
        "",
        f"- mining_id_count: {stats['mining_id_count']}",
        f"- hard_negative_count: {stats['hard_negative_count']}",
        f"- heldout_leakage_count: {stats['heldout_leakage_count']}",
        f"- uses_model_predictions: {str(stats['uses_model_predictions']).lower()}",
        "",
        "## Hard Types",
    ]
    lines += [f"- {key}: {value}" for key, value in sorted(stats["hard_type_counts"].items())]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine r3 actual control failures for Preference-v8.1.")
    parser.add_argument("--prediction_root", default="outputs/predictions_mining")
    parser.add_argument("--prediction_dir", default="", help="Direct strict_visual prediction directory containing the four setting JSONL files.")
    parser.add_argument("--model_name", default="sft_v3_r3_lingo_smoke")
    parser.add_argument("--dataset", default="lingoqa")
    parser.add_argument("--mode", default="strict_visual")
    parser.add_argument("--mining_ids", default="data/mining/v8_1/lingoqa_mining_pool_ids.json")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--output", default="data/mining/v8_1/hard_negatives_v8_1.jsonl")
    parser.add_argument("--stats", default="data/mining/v8_1/hard_negatives_v8_1_stats.json")
    parser.add_argument("--report", default="outputs/data_audit/hard_negatives_v8_1_report.md")
    parser.add_argument("--output_dir", default="", help="Compatibility alias controlling hard-negative JSONL/stats paths.")
    parser.add_argument("--run", action="store_true", help="Accepted for Stage 7 orchestration compatibility; mining is offline only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        args.output = str(output_dir / "hard_negatives_v8_1.jsonl")
        args.stats = str(output_dir / "hard_negatives_v8_1_stats.json")
    expected_ids = read_ids(Path(args.mining_ids))
    _, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    try:
        if args.prediction_dir:
            prediction_dir = Path(args.prediction_dir)
            aligned = align_settings({setting: read_jsonl(prediction_dir / f"{setting}.jsonl") for setting in SETTINGS})
        else:
            aligned = align_settings(load_model_settings(Path(args.prediction_root), args.dataset, args.model_name, args.mode))
    except FileNotFoundError as exc:
        raise SystemExit(f"r3 mining predictions have not been run yet: {exc}") from None
    rows, stats = mine(aligned, expected_ids, heldout_keys)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.stats), stats)
    write_markdown(Path(args.report), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
