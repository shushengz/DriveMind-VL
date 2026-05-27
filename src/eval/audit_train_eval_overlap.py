"""Audit normalized train/eval ID overlap for Stage 4.5.

This module is offline-only: it reads existing JSONL artifacts and never loads
or invokes a model.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")
TRAIN_SUFFIXES = (
    "normal_replay",
    "spatial_normal_qa",
    "extra_normal_fill",
    "blank_calibration",
    "wrong_image_calibration",
    "text_only_calibration",
    "normal_visual_qa",
    "spatial_hard_negative",
    "blank_refusal",
    "wrong_image_caution",
    "text_only_caution",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing JSONL input: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def normalize_id(sample_id: Any, row: dict[str, Any] | None = None) -> str:
    """Return source-level ID, stripping generated calibration suffixes."""
    if row:
        source_id = (row.get("metadata") or {}).get("source_id")
        if source_id:
            return str(source_id)
    value = str(sample_id or "")
    for suffix in sorted(TRAIN_SUFFIXES, key=len, reverse=True):
        marker = "_" + suffix
        if value.endswith(marker):
            return value[: -len(marker)]
    return value


def normalized_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {normalize_id(row.get("id"), row) for row in rows if normalize_id(row.get("id"), row)}


def audit_overlap(train_file: Path, prediction_dir: Path) -> dict[str, Any]:
    train_rows = read_jsonl(train_file)
    train_ids = normalized_ids(train_rows)
    setting_results: dict[str, Any] = {}
    eval_ids: set[str] = set()
    for setting in SETTINGS:
        path = prediction_dir / f"{setting}.jsonl"
        setting_ids = normalized_ids(read_jsonl(path))
        overlap = sorted(train_ids & setting_ids)
        setting_results[setting] = {
            "eval_unique_ids": len(setting_ids),
            "overlap_count": len(overlap),
            "overlap_rate_eval": len(overlap) / len(setting_ids) if setting_ids else 0.0,
            "overlap_ids": overlap,
        }
        eval_ids.update(setting_ids)

    overlap_ids = sorted(train_ids & eval_ids)
    overlap_rate_eval = len(overlap_ids) / len(eval_ids) if eval_ids else 0.0
    overlap_rate_train = len(overlap_ids) / len(train_ids) if train_ids else 0.0
    exists = bool(overlap_ids)
    return {
        "train_file": train_file.as_posix(),
        "prediction_dir": prediction_dir.as_posix(),
        "train_unique_ids": len(train_ids),
        "eval_unique_ids": len(eval_ids),
        "overlap_ids": overlap_ids,
        "overlap_count": len(overlap_ids),
        "overlap_rate_eval": overlap_rate_eval,
        "overlap_rate_train": overlap_rate_train,
        "overlap_exists": exists,
        "stage4_results_may_be_affected_by_leakage": exists,
        "heldout_eval_required": exists,
        "by_setting": setting_results,
    }


def write_outputs(result: dict[str, Any], output_json: Path, output_csv: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = ["id", "in_train", "in_eval", *SETTINGS]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        by_setting = {setting: set(result["by_setting"][setting]["overlap_ids"]) for setting in SETTINGS}
        for sample_id in result["overlap_ids"]:
            writer.writerow(
                {"id": sample_id, "in_train": True, "in_eval": True, **{setting: sample_id in by_setting[setting] for setting in SETTINGS}}
            )

    yes = "\u662f"
    no = "\u5426"
    affected = yes if result["stage4_results_may_be_affected_by_leakage"] else no
    required = yes if result["heldout_eval_required"] else no
    lines = [
        "# Stage 4.5 Train/Eval Leakage Audit",
        "",
        f"- \u662f\u5426\u5b58\u5728 train/eval overlap\uff1a{yes if result['overlap_exists'] else no}",
        f"- train unique IDs\uff1a{result['train_unique_ids']}",
        f"- eval unique IDs\uff1a{result['eval_unique_ids']}",
        f"- overlap count\uff1a{result['overlap_count']}",
        f"- overlap rate (eval)\uff1a{result['overlap_rate_eval']:.2%}",
        f"- overlap rate (train)\uff1a{result['overlap_rate_train']:.2%}",
        f"- Stage 4 \u7ed3\u679c\u662f\u5426\u53ef\u80fd\u53d7 leakage \u5f71\u54cd\uff1a{affected}",
        f"- \u662f\u5426\u9700\u8981 held-out eval\uff1a{required}",
        "",
        "## Settings",
    ]
    for setting in SETTINGS:
        item = result["by_setting"][setting]
        lines.append(f"- {setting}: {item['overlap_count']}/{item['eval_unique_ids']} ({item['overlap_rate_eval']:.2%})")
    if result["overlap_exists"]:
        lines += [
            "",
            "## \u7ed3\u8bba",
            "",
            "Stage 4 r3 \u8bc4\u6d4b\u6837\u672c\u4e0e\u8bad\u7ec3\u6765\u6e90 ID \u5b58\u5728\u91cd\u53e0\uff0c\u539f\u59cb\u6027\u80fd\u6570\u5b57\u53ef\u80fd\u88ab\u9ad8\u4f30\u3002\u5728\u771f\u6b63\u672a\u53c2\u4e0e\u8bad\u7ec3\u7684 held-out IDs \u4e0a\u5b8c\u6210\u590d\u8bc4\u524d\uff0c\u4e0d\u5e94\u636e\u6b64\u8fdb\u5165 DPO-v8\u3002",
        ]
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit r3 training ID overlap with Stage 4 eval predictions.")
    parser.add_argument("--train_file", default="data/train/sft_v3_r3/lingoqa_sft_v3_r3.jsonl")
    parser.add_argument("--prediction_dir", default="outputs/predictions/lingoqa/sft_v3_r3_lingo_smoke/strict_visual")
    parser.add_argument("--output_json", default="outputs/final_report/stage4_5_train_eval_overlap.json")
    parser.add_argument("--output_csv", default="outputs/final_report/stage4_5_train_eval_overlap.csv")
    parser.add_argument("--output_md", default="outputs/final_report/stage4_5_train_eval_overlap.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_overlap(Path(args.train_file), Path(args.prediction_dir))
    write_outputs(result, Path(args.output_json), Path(args.output_csv), Path(args.output_md))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
