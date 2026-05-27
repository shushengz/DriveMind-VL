"""Filter candidate DPO training rows against Stage 4.5 held-out identities.

The held-out file is used only as an exclusion list. This module performs no
model loading, inference, or training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SAMPLE_SUFFIXES = (
    "normal_replay",
    "spatial_normal_qa",
    "extra_normal_fill",
    "blank_calibration",
    "wrong_image_calibration",
    "text_only_calibration",
    "normal_anchor_gold_vs_bad",
    "normal_gold_vs_refusal",
    "control_abstain_vs_hallucination",
    "wrong_image_caution_vs_confident_answer",
    "spatial_gold_vs_spatial_wrong",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing JSONL input: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_id(value: Any) -> str:
    sample_id = str(value or "")
    for suffix in sorted(SAMPLE_SUFFIXES, key=len, reverse=True):
        marker = "_" + suffix
        if sample_id.endswith(marker):
            sample_id = sample_id[: -len(marker)]
            break
    return sample_id


def identity_keys(value: Any, row: dict[str, Any] | None = None) -> set[str]:
    sample_id = normalize_id(value)
    keys = {sample_id} if sample_id else set()
    if sample_id.startswith("lingoqa_test_"):
        keys.add(sample_id[len("lingoqa_test_") :])
    metadata = (row or {}).get("metadata") or {}
    for key in ("source_id", "source_original_id", "original_id"):
        if metadata.get(key):
            keys.add(normalize_id(metadata[key]))
    external = (((row or {}).get("meta") or {}).get("external") or {})
    if external.get("question_id"):
        keys.add(str(external["question_id"]))
    return {key for key in keys if key}


def read_heldout_keys(path: Path) -> tuple[list[str], set[str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing held-out ID input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list):
        raise ValueError("held-out ID input must be a list or an object with an 'ids' list")
    keys: set[str] = set()
    for sample_id in ids:
        keys.update(identity_keys(sample_id))
    return [str(sample_id) for sample_id in ids], keys


def filter_candidates(rows: list[dict[str, Any]], heldout_keys: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, removed = [], []
    for row in rows:
        if identity_keys(row.get("id"), row) & heldout_keys:
            removed.append(row)
        else:
            kept.append(row)
    return kept, removed


def build_report(rows: list[dict[str, Any]], kept: list[dict[str, Any]], removed: list[dict[str, Any]], heldout_ids: list[str], heldout_keys: set[str]) -> dict[str, Any]:
    leakage_after = [str(row.get("id", "")) for row in kept if identity_keys(row.get("id"), row) & heldout_keys]
    return {
        "candidate_count": len(rows),
        "heldout_id_count": len(heldout_ids),
        "removed_count": len(removed),
        "removed_ids": [str(row.get("id", "")) for row in removed],
        "remaining_count": len(kept),
        "leakage_after_filter": len(leakage_after),
        "leakage_after_filter_ids": leakage_after,
        "train_ready": len(leakage_after) == 0,
    }


def write_report(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DPO-v8 Held-out Exclusion Audit",
        "",
        f"- candidate_count: {report['candidate_count']}",
        f"- heldout_id_count: {report['heldout_id_count']}",
        f"- removed_count: {report['removed_count']}",
        f"- remaining_count: {report['remaining_count']}",
        f"- leakage_after_filter: {report['leakage_after_filter']}",
        f"- train_ready: {str(report['train_ready']).lower()}",
        "",
        "Held-out IDs are used only for exclusion and are not a source of DPO-v8 pairs.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exclude Stage 4.5 held-out IDs from DPO-v8 candidate rows.")
    parser.add_argument("--candidate_file", default="data/processed/visual_control/lingoqa_strict_normal.jsonl")
    parser.add_argument("--heldout_ids", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--filtered_output", default="data/train/preference_v8/eligible_normal_pool.jsonl")
    parser.add_argument("--output_json", default="outputs/data_audit/dpo_v8_heldout_exclusion.json")
    parser.add_argument("--output_md", default="outputs/data_audit/dpo_v8_heldout_exclusion.md")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(Path(args.candidate_file))
    if args.dry_run:
        rows = rows[: min(32, len(rows))]
    heldout_ids, heldout_keys = read_heldout_keys(Path(args.heldout_ids))
    kept, removed = filter_candidates(rows, heldout_keys)
    report = build_report(rows, kept, removed, heldout_ids, heldout_keys)
    write_jsonl(Path(args.filtered_output), kept)
    write_report(report, Path(args.output_json), Path(args.output_md))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["train_ready"]:
        raise SystemExit("held-out leakage remains after filtering; DPO-v8 construction is blocked")


if __name__ == "__main__":
    main()
