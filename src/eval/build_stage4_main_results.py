
"""Build Stage 4 main results from existing predictions.

No model loading or inference is performed. Missing r3 predictions are allowed
so CPU-only preparation can run before GPU training.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.metrics_visual_control import SETTINGS
from src.eval.rescore_answer_only import (
    DEFAULT_MODELS,
    MAIN_COLUMNS,
    align_settings,
    load_model_settings,
    parse_models,
    summarize_aligned,
    write_csv,
)


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--prediction_root", default="outputs/predictions")
    p.add_argument("--dataset", default="lingoqa")
    p.add_argument("--mode", default="strict_visual")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS))
    p.add_argument("--output", default="outputs/final_report/stage4_sft_v3_r3_main_results.csv")
    p.add_argument("--warnings_output", default="outputs/final_report/stage4_sft_v3_r3_main_results_warnings.json")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def case_id(group):
    return str(group["normal"].get("id", ""))


def filter_to_common_ids(aligned, common_ids, ordered_ids):
    by_id={case_id(group): group for group in aligned}
    return [by_id[sid] for sid in ordered_ids if sid in common_ids and sid in by_id]


def build_common_id_rows(prediction_root: Path, dataset: str, mode: str, models: list[str], dry_run: bool):
    warnings=[]
    aligned_by_model={}
    for model in models:
        try:
            aligned=align_settings(load_model_settings(prediction_root, dataset, model, mode))
        except Exception as exc:
            warnings.append(f"skip {model}: {exc}")
            continue
        if not aligned:
            warnings.append(f"skip {model}: no aligned cases")
            continue
        aligned_by_model[model]=aligned
    if not aligned_by_model:
        return [], warnings

    id_sets={model: {case_id(group) for group in aligned} for model, aligned in aligned_by_model.items()}
    common_ids=set.intersection(*id_sets.values())
    if not common_ids:
        raise ValueError(f"no common ids across available models: {sorted(aligned_by_model)}")

    first_model=next(iter(aligned_by_model))
    ordered_ids=[case_id(group) for group in aligned_by_model[first_model] if case_id(group) in common_ids]
    if dry_run:
        ordered_ids=ordered_ids[: min(8, len(ordered_ids))]
        common_ids=set(ordered_ids)

    for model, ids in id_sets.items():
        if len(ids) != len(common_ids):
            warnings.append(f"{model}: using {len(common_ids)} common ids out of {len(ids)} aligned cases")

    rows=[]
    for model, aligned in aligned_by_model.items():
        model_aligned=filter_to_common_ids(aligned, common_ids, ordered_ids)
        for score_mode in ("raw_full", "answer_only"):
            rows.append(summarize_aligned(model_aligned, model, dataset, mode, score_mode, with_ci=True))
    return rows, warnings


def main():
    args=parse_args()
    rows,warnings=build_common_id_rows(Path(args.prediction_root), args.dataset, args.mode, parse_models(args.models), args.dry_run)
    write_csv(Path(args.output), rows, MAIN_COLUMNS)
    Path(args.warnings_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.warnings_output).write_text(json.dumps({"warnings": warnings}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": args.output, "rows": len(rows), "warnings": warnings}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
