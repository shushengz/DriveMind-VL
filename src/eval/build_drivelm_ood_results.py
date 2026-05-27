"""Build Stage 13 DriveLM OOD strict visual-control metric tables."""
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

DEFAULT_MODELS = ["base_qwen25vl_3b", "sft_v3_r3_lingo_smoke"]
COLUMNS = [
    "model_name", "dataset", "eval_split", "num_samples", "score_mode",
    "normal_f1", "text_only_f1", "wrong_image_f1", "blank_image_f1",
    "setting_gap", "case_gap", "positive_gap_rate",
    "control_hallucination_rate", "control_direct_answer_rate",
    "control_high_f1_rate_0_20", "blank_high_f1_rate_0_20",
    "normal_refusal_rate", "avg_answer_length",
]


def behavior(aligned: list[dict[str, dict[str, Any]]], score_mode: str) -> dict[str, float]:
    controls: dict[str, list[dict[str, Any]]] = {setting: [] for setting in CONTROL_SETTINGS}
    for group in aligned:
        for setting in CONTROL_SETTINGS:
            score = score_prediction(group[setting], score_mode)
            controls[setting].append({**score, **behavior_flags(score["score_text"], score["f1"])})
    rows = [row for setting in CONTROL_SETTINGS for row in controls[setting]]
    return {
        "control_direct_answer_rate": mean([float(row["is_direct_answer"]) for row in rows]),
        "control_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in rows]),
        "blank_high_f1_rate_0_20": mean([float(row["high_f1_0_20"]) for row in controls["blank_image"]]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 13 DriveLM OOD results.")
    parser.add_argument("--prediction_root", default="outputs/predictions_drivelm_ood")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--eval_split", default="drivelm_ood_100")
    parser.add_argument("--output", default="outputs/final_report/drivelm_ood_results_100.csv")
    parser.add_argument("--answer_only_output", default="outputs/final_report/drivelm_ood_results_100_answer_only.csv")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    expected_n: int | None = None
    for model in [name.strip() for name in args.models.split(",") if name.strip()]:
        aligned = align_settings(load_model_settings(Path(args.prediction_root), "drivelm", model, "strict_visual"))
        if expected_n is None:
            expected_n = len(aligned)
        if len(aligned) != expected_n:
            raise ValueError(f"prediction count mismatch: {model} has {len(aligned)} cases, expected {expected_n}")
        for score_mode in ("raw_full", "answer_only"):
            result = summarize_aligned(aligned, model, "drivelm", "strict_visual", score_mode, with_ci=False)
            result.update(behavior(aligned, score_mode))
            result["eval_split"] = args.eval_split
            rows.append(result)

    if expected_n != 100 and args.eval_split.endswith("_100"):
        raise ValueError(f"100-case result build expected exactly 100 aligned cases, found {expected_n}")
    write_csv(Path(args.output), rows)
    write_csv(Path(args.answer_only_output), [row for row in rows if row["score_mode"] == "answer_only"])
    print(json.dumps({"rows": len(rows), "num_samples": expected_n, "output": args.output, "answer_only": args.answer_only_output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
