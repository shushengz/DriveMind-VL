
"""Rescore existing visual-control predictions with raw-full and answer-only modes."""
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

from src.eval.metrics_visual_control import CONTROL_SETTINGS, SETTINGS, answer_length, bootstrap_ci, exact_match, is_control_hallucination, is_refusal, mean, token_f1
from src.eval.prediction_parser import parse_prediction

DEFAULT_MODELS = ["base_qwen25vl_3b", "sft_v2", "dpo_v7", "sft_v3_lingo_smoke", "sft_v3_r2_lingo_smoke", "sft_v3_r3_lingo_smoke"]
OUTPUT_COLUMNS = ["model_name", "dataset", "mode", "score_mode", "num_samples", "normal_f1", "text_only_f1", "wrong_image_f1", "blank_image_f1", "setting_gap", "case_gap", "positive_gap_rate", "control_hallucination_rate", "normal_refusal_rate", "avg_answer_length", "parse_success_rate", "json_format_rate"]
MAIN_COLUMNS = OUTPUT_COLUMNS[:15] + ["normal_f1_ci_low", "normal_f1_ci_high", "case_gap_ci_low", "case_gap_ci_high"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    if not path.exists():
        raise FileNotFoundError(path.as_posix())
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def load_model_settings(prediction_root: Path, dataset: str, model: str, mode: str) -> dict[str, list[dict[str, Any]]]:
    base = prediction_root / dataset / model / mode
    return {setting: read_jsonl(base / f"{setting}.jsonl") for setting in SETTINGS}


def align_settings(rows_by_setting: dict[str, list[dict[str, Any]]]) -> list[dict[str, dict[str, Any]]]:
    maps={setting: {str(row.get("id")): row for row in rows if row.get("id")} for setting, rows in rows_by_setting.items()}
    common=set(maps["normal"])
    for setting in SETTINGS[1:]:
        common &= set(maps[setting])
    if not common:
        counts={setting: len(rows_by_setting.get(setting, [])) for setting in SETTINGS}
        raise ValueError(f"no common ids across settings; counts={counts}")
    missing_messages=[]
    for setting in SETTINGS:
        missing=sorted(set(maps["normal"]) - set(maps[setting]))[:5]
        if missing:
            missing_messages.append(f"{setting} missing examples: {missing}")
    if missing_messages:
        raise ValueError("settings are not aligned: " + "; ".join(missing_messages))
    ordered=[str(row.get("id")) for row in rows_by_setting["normal"] if str(row.get("id")) in common]
    return [{setting: maps[setting][sid] for setting in SETTINGS} for sid in ordered]


def score_prediction(row: dict[str, Any], score_mode: str) -> dict[str, Any]:
    prediction = row.get("prediction", "")
    parsed = parse_prediction(prediction)
    text = parsed["answer_text"] if score_mode == "answer_only" else prediction
    gold = row.get("gold", "")
    setting = str(row.get("setting") or "")
    return {
        "id": row.get("id", ""),
        "setting": setting,
        "gold": gold,
        "prediction": prediction,
        "score_text": text,
        "f1": token_f1(text, gold),
        "em": exact_match(text, gold),
        "refusal": is_refusal(text),
        "hallucination": is_control_hallucination(text, setting),
        "answer_length": answer_length(text),
        "parse_success": parsed["parse_success"],
        "format_type": parsed["format_type"],
    }


def summarize_aligned(aligned: list[dict[str, dict[str, Any]]], model: str, dataset: str, mode: str, score_mode: str, with_ci: bool = False) -> dict[str, Any]:
    cases=[]
    all_scored=[]
    for group in aligned:
        scored={setting: score_prediction(group[setting], score_mode) for setting in SETTINGS}
        all_scored.extend(scored.values())
        control_f1={setting: scored[setting]["f1"] for setting in CONTROL_SETTINGS}
        best=max(control_f1, key=control_f1.get)
        normal_f1=scored["normal"]["f1"]
        cases.append({"normal_f1": normal_f1, "case_gap": normal_f1 - control_f1[best]})
    by_setting={setting: [score_prediction(group[setting], score_mode) for group in aligned] for setting in SETTINGS}
    setting_f1={setting: mean([row["f1"] for row in rows]) for setting, rows in by_setting.items()}
    case_gaps=[case["case_gap"] for case in cases]
    control_rows=[row for setting in CONTROL_SETTINGS for row in by_setting[setting]]
    parse_rate=mean([1.0 if row["parse_success"] else 0.0 for row in all_scored])
    json_rate=mean([1.0 if row["format_type"] in {"json", "json_like"} else 0.0 for row in all_scored])
    out={
        "model_name": model,
        "dataset": dataset,
        "mode": mode,
        "score_mode": score_mode,
        "num_samples": len(aligned),
        "normal_f1": setting_f1["normal"],
        "text_only_f1": setting_f1["text_only"],
        "wrong_image_f1": setting_f1["wrong_image"],
        "blank_image_f1": setting_f1["blank_image"],
        "setting_gap": setting_f1["normal"] - max(setting_f1[s] for s in CONTROL_SETTINGS),
        "case_gap": mean(case_gaps),
        "positive_gap_rate": mean([1.0 if gap > 0 else 0.0 for gap in case_gaps]),
        "control_hallucination_rate": mean([1.0 if row["hallucination"] else 0.0 for row in control_rows]),
        "normal_refusal_rate": mean([1.0 if row["refusal"] else 0.0 for row in by_setting["normal"]]),
        "avg_answer_length": mean([float(row["answer_length"]) for row in all_scored]),
        "parse_success_rate": parse_rate,
        "json_format_rate": json_rate,
    }
    if with_ci:
        nci=bootstrap_ci([row["f1"] for row in by_setting["normal"]])
        cgci=bootstrap_ci(case_gaps)
        out.update({"normal_f1_ci_low": nci[0], "normal_f1_ci_high": nci[1], "case_gap_ci_low": cgci[0], "case_gap_ci_high": cgci[1]})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer=csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def parse_models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_rows(prediction_root: Path, dataset: str, mode: str, models: list[str], dry_run: bool = False, with_ci: bool = False, allow_missing: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    rows=[]; warnings=[]
    for model in models:
        try:
            aligned=align_settings(load_model_settings(prediction_root, dataset, model, mode))
        except Exception as exc:
            msg=f"skip {model}: {exc}"
            if allow_missing:
                warnings.append(msg); continue
            raise
        if dry_run:
            aligned=aligned[: min(8, len(aligned))]
        for score_mode in ("raw_full", "answer_only"):
            rows.append(summarize_aligned(aligned, model, dataset, mode, score_mode, with_ci=with_ci))
    return rows, warnings


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="Rescore visual-control predictions using answer-only parser.")
    p.add_argument("--prediction_root", default="outputs/predictions")
    p.add_argument("--dataset", default="lingoqa")
    p.add_argument("--mode", default="strict_visual")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS[:-1]))
    p.add_argument("--output", default="outputs/final_report/stage4_answer_only_rescore.csv")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--allow_missing", action="store_true")
    return p.parse_args()


def main() -> None:
    args=parse_args()
    rows, warnings = build_rows(Path(args.prediction_root), args.dataset, args.mode, parse_models(args.models), args.dry_run, with_ci=False, allow_missing=args.allow_missing)
    write_csv(Path(args.output), rows, OUTPUT_COLUMNS)
    print(json.dumps({"output": args.output, "rows": len(rows), "warnings": warnings}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
