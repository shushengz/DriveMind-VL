"""Build aligned offline reward-evaluation records from saved predictions only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.rescore_answer_only import align_settings, load_model_settings, score_prediction

SOURCES = [
    ("lingoqa", "outputs/predictions_heldout", "sft_v3_r3_lingo_smoke", "SFT-v3-r3"),
    ("lingoqa", "outputs/predictions_heldout", "dpo_v8_lingo_smoke", "DPO-v8"),
    ("lingoqa", "outputs/predictions_heldout", "dpo_v8_1_lingo_smoke_step25", "DPO-v8.1 step-25"),
    ("lingoqa", "outputs/predictions_heldout", "dpo_v8_2_lingo_smoke_step25", "DPO-v8.2 step-25"),
    ("drivelm", "outputs/predictions_drivelm_ood", "base_qwen25vl_3b", "Base Qwen2.5-VL-3B"),
    ("drivelm", "outputs/predictions_drivelm_ood", "sft_v3_r3_lingo_smoke", "SFT-v3-r3"),
]


def load_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build records for offline GRPO-lite reward v2 evaluation.")
    parser.add_argument("--output", default="outputs/final_report/reward_v2_eval_records.jsonl")
    parser.add_argument("--summary", default="outputs/final_report/reward_v2_eval_records_summary.json")
    parser.add_argument("--taxonomy", default="outputs/final_report/drivelm_ood_failure_taxonomy.csv")
    args = parser.parse_args()
    taxonomy = load_taxonomy(Path(args.taxonomy))
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    for dataset, prediction_root, internal_name, display_name in SOURCES:
        key = f"{dataset}:{display_name}"
        try:
            aligned = align_settings(load_model_settings(Path(prediction_root), dataset, internal_name, "strict_visual"))
        except Exception as exc:
            warnings.append(f"missing {key}: {exc}")
            continue
        for group in aligned:
            sample_id = str(group["normal"].get("id", ""))
            extra = taxonomy.get(sample_id, {}) if dataset == "drivelm" and internal_name == "sft_v3_r3_lingo_smoke" else {}
            capability = extra.get("capability", "") if dataset == "drivelm" else ""
            if dataset == "drivelm" and not capability:
                # Base receives capability context but not r3-derived failure tags.
                capability = taxonomy.get(sample_id, {}).get("capability", "")
            failure_tags = extra.get("failure_tags", "").split("|") if extra.get("failure_tags") else []
            scores = {setting: score_prediction(row, "answer_only") for setting, row in group.items()}
            records.append({
                "id": sample_id,
                "dataset": dataset,
                "model_name": display_name,
                "internal_model_name": internal_name,
                "question": group["normal"].get("question", ""),
                "gold": group["normal"].get("gold", ""),
                "capability": capability,
                "failure_tags": failure_tags,
                "image_labels": group["normal"].get("image_labels", []) if dataset == "drivelm" else [],
                "settings": group,
                "answer_only_f1": {setting: float(score["f1"]) for setting, score in scores.items()},
                "case_gap": float(scores["normal"]["f1"]) - max(float(scores[setting]["f1"]) for setting in ("text_only", "wrong_image", "blank_image")),
                "usage": "offline_reward_evaluation_only_not_training",
            })
        counts[key] = len(aligned)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "total_records": len(records),
        "counts": counts,
        "warnings": warnings,
        "training_use_forbidden": True,
        "answer_only_parser_used": True,
        "drivelm_records_have_image_labels": all(record["image_labels"] for record in records if record["dataset"] == "drivelm"),
    }
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
