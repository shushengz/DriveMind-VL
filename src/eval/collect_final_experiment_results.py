"""Collect final held-out strict visual-control results from saved predictions only."""
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

from src.eval.control_behavior_metrics import behavior_flags
from src.eval.metrics_visual_control import CONTROL_SETTINGS, mean
from src.eval.rescore_answer_only import align_settings, load_model_settings, score_prediction, summarize_aligned

MODELS = [
    ("base_qwen25vl_3b", "Base Qwen2.5-VL-3B", "Baseline", "base", "Pretrained baseline."),
    ("sft_v2", "SFT-v2", "Earlier", "sft", "Earlier supervised calibration baseline."),
    ("dpo_v7", "DPO-v7", "Earlier", "dpo", "Earlier preference baseline."),
    ("sft_v3_r2_lingo_smoke", "SFT-v3-r2", "Stage 3", "sft", "Control-heavy SFT calibration ablation."),
    ("sft_v3_r3_lingo_smoke", "SFT-v3-r3", "Stage 4.5", "sft", "Selected final candidate."),
    ("dpo_v8_lingo_smoke", "DPO-v8 rule-based", "Stage 6", "dpo", "Rule-based preference ablation."),
    ("dpo_v8_1_lingo_smoke_step25", "DPO-v8.1 step-25", "Stage 8", "dpo", "Model-mined preference ablation."),
    ("dpo_v8_1_lingo_smoke_step50", "DPO-v8.1 step-50", "Stage 8", "dpo", "Longer model-mined preference ablation."),
    ("dpo_v8_2_lingo_smoke_step25", "DPO-v8.2 step-25", "Stage 10", "dpo", "Case-gap-aware preference ablation."),
]
COLUMNS = [
    "model_name", "stage", "method_type", "dataset", "eval_split", "num_samples", "score_mode",
    "normal_f1", "text_only_f1", "wrong_image_f1", "blank_image_f1", "setting_gap", "case_gap",
    "positive_gap_rate", "control_hallucination_rate", "control_direct_answer_rate",
    "control_high_f1_rate_0_20", "blank_high_f1_rate_0_20", "normal_refusal_rate",
    "avg_answer_length", "is_leaked_eval", "is_final_candidate", "comment",
]


def load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or len(ids) != 100 or len(set(ids)) != 100:
        raise ValueError(f"held-out IDs must contain 100 unique IDs: {path}")
    return [str(item) for item in ids]


def align_to_ids(aligned: list[dict[str, dict[str, Any]]], ids: list[str], model: str) -> list[dict[str, dict[str, Any]]]:
    by_id = {str(group["normal"].get("id")): group for group in aligned}
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    extra = sorted(set(by_id) - set(ids))
    if missing or extra:
        raise ValueError(f"{model} held-out mismatch; missing={missing[:5]} extra={extra[:5]}")
    return [by_id[sample_id] for sample_id in ids]


def behavior_metrics(aligned: list[dict[str, dict[str, Any]]], score_mode: str) -> dict[str, float]:
    per_setting: dict[str, list[dict[str, Any]]] = {setting: [] for setting in CONTROL_SETTINGS}
    for group in aligned:
        for setting in CONTROL_SETTINGS:
            score = score_prediction(group[setting], score_mode)
            per_setting[setting].append({**score, **behavior_flags(score["score_text"], score["f1"])})
    control_rows = [row for setting in CONTROL_SETTINGS for row in per_setting[setting]]
    return {
        "control_direct_answer_rate": mean([float(row["is_direct_answer"]) for row in control_rows]),
        "control_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in control_rows]),
        "blank_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in per_setting["blank_image"]]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in COLUMNS} for row in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect final held-out result tables without inference.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--eval_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--dataset", default="lingoqa")
    parser.add_argument("--mode", default="strict_visual")
    parser.add_argument("--output_dir", default="outputs/final_report")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    ids = load_ids(Path(args.eval_ids))
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for internal_name, display_name, stage, method_type, comment in MODELS:
        aligned = align_settings(load_model_settings(Path(args.prediction_root), args.dataset, internal_name, args.mode))
        aligned = align_to_ids(aligned, ids, internal_name)
        if len(aligned) != 100:
            warnings.append(f"{display_name}: expected 100 held-out rows, found {len(aligned)}")
        for score_mode in ("raw_full", "answer_only"):
            metrics = summarize_aligned(aligned, display_name, args.dataset, args.mode, score_mode, with_ci=False)
            metrics.update(behavior_metrics(aligned, score_mode))
            metrics.update({
                "model_name": display_name,
                "stage": stage,
                "method_type": method_type,
                "eval_split": "heldout_strict_visual_100",
                "is_leaked_eval": False,
                "is_final_candidate": display_name == "SFT-v3-r3",
                "comment": comment,
            })
            rows.append(metrics)
    discovered = sorted(path.name for path in Path("outputs/final_report").glob("*main_results.csv"))
    excluded_leaked = [name for name in discovered if name.startswith("stage4_") and not name.startswith("stage4_5_")]
    metadata = {
        "source": "saved held-out strict_visual predictions only",
        "heldout_ids": args.eval_ids,
        "expected_samples": 100,
        "models": [item[1] for item in MODELS],
        "warnings": warnings,
        "discovered_result_tables": discovered,
        "excluded_as_pre_heldout_or_leaked": excluded_leaked,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "final_main_results_raw.csv", [row for row in rows if row["score_mode"] == "raw_full"])
    write_csv(output_dir / "final_main_results_answer_only.csv", [row for row in rows if row["score_mode"] == "answer_only"])
    (output_dir / "final_main_results_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "answer_only_rows": 9, "warnings": warnings, "excluded_leaked": excluded_leaked}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
