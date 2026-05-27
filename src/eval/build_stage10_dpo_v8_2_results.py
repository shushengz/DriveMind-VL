"""Build the Stage 10 held-out table including case-gap behavior metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.build_stage4_main_results import build_common_id_rows
from src.eval.control_behavior_metrics import behavior_flags
from src.eval.metrics_visual_control import CONTROL_SETTINGS, mean
from src.eval.rescore_answer_only import align_settings, load_model_settings, score_prediction, write_csv

MODELS = [
    "base_qwen25vl_3b",
    "sft_v2",
    "dpo_v7",
    "sft_v3_r3_lingo_smoke",
    "dpo_v8_lingo_smoke",
    "dpo_v8_1_lingo_smoke_step25",
    "dpo_v8_1_lingo_smoke_step50",
    "dpo_v8_2_lingo_smoke_step25",
]
COLUMNS = [
    "model_name", "dataset", "num_samples", "score_mode", "normal_f1",
    "text_only_f1", "wrong_image_f1", "blank_image_f1", "setting_gap",
    "case_gap", "positive_gap_rate", "control_hallucination_rate",
    "normal_refusal_rate", "control_direct_answer_rate",
    "control_high_f1_rate_0_20", "blank_high_f1_rate_0_20",
    "text_only_high_f1_rate_0_20", "wrong_image_high_f1_rate_0_20",
    "avg_answer_length", "normal_f1_ci_low", "normal_f1_ci_high",
    "case_gap_ci_low", "case_gap_ci_high",
]


def behavior_metrics(root: Path, dataset: str, model: str, mode: str, score_mode: str) -> dict[str, float]:
    aligned = align_settings(load_model_settings(root, dataset, model, mode))
    rows_by_setting: dict[str, list[dict[str, Any]]] = {setting: [] for setting in CONTROL_SETTINGS}
    for group in aligned:
        for setting in CONTROL_SETTINGS:
            score = score_prediction(group[setting], score_mode)
            rows_by_setting[setting].append({**score, **behavior_flags(score["score_text"], score["f1"])})
    control = [row for setting in CONTROL_SETTINGS for row in rows_by_setting[setting]]
    return {
        "control_direct_answer_rate": mean([float(row["is_direct_answer"]) for row in control]),
        "control_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in control]),
        "blank_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in rows_by_setting["blank_image"]]),
        "text_only_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in rows_by_setting["text_only"]]),
        "wrong_image_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in rows_by_setting["wrong_image"]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Stage 10 DPO-v8.2 held-out results.")
    parser.add_argument("--prediction_root", default="outputs/predictions_heldout")
    parser.add_argument("--dataset", default="lingoqa")
    parser.add_argument("--mode", default="strict_visual")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--output", default="outputs/final_report/stage10_dpo_v8_2_heldout_main_results.csv")
    parser.add_argument("--warnings_output", default="outputs/final_report/stage10_dpo_v8_2_heldout_main_results_warnings.json")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    rows, warnings = build_common_id_rows(Path(args.prediction_root), args.dataset, args.mode, models, args.dry_run)
    for row in rows:
        try:
            row.update(behavior_metrics(Path(args.prediction_root), args.dataset, row["model_name"], args.mode, row["score_mode"]))
        except Exception as exc:
            warnings.append(f"behavior metrics unavailable for {row['model_name']}/{row['score_mode']}: {exc}")
    write_csv(Path(args.output), rows, COLUMNS)
    Path(args.warnings_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.warnings_output).write_text(json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "rows": len(rows), "warnings": warnings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
