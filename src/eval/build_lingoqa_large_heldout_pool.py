"""Build leakage-filtered LingoQA held-out pools from existing strict controls."""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")
SETTING_SUFFIX = re.compile(r"_(normal|text_only|wrong_image|blank_image)(?:_[A-Za-z0-9_]+)?$")
SAMPLE_SUFFIXES = (
    "normal_replay", "normal_anchor", "spatial_normal_qa", "extra_normal_fill",
    "blank_calibration", "wrong_image_calibration", "text_only_calibration",
    "normal_visual_qa", "spatial_hard_negative", "blank_refusal",
    "wrong_image_caution", "text_only_caution",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_id(value: Any, row: dict[str, Any] | None = None) -> str:
    if row:
        direct = row.get("source_id") or (row.get("metadata") or {}).get("source_id")
        if direct:
            return normalize_id(direct)
    sample_id = str(value or "")
    for suffix in sorted(SAMPLE_SUFFIXES, key=len, reverse=True):
        marker = "_" + suffix
        if sample_id.endswith(marker):
            sample_id = sample_id[:-len(marker)]
            break
    sample_id = SETTING_SUFFIX.sub("", sample_id)
    return sample_id


def aligned_rows(dirs: list[Path], dataset: str) -> tuple[dict[str, dict[str, dict[str, Any]]], int, int]:
    by_setting: dict[str, dict[str, dict[str, Any]]] = {setting: {} for setting in SETTINGS}
    source_count = 0
    for directory in dirs:
        for setting in SETTINGS:
            for row in read_jsonl(directory / f"{dataset}_strict_{setting}.jsonl"):
                source_count += setting == "normal"
                sample_id = normalize_id(row.get("id"), row)
                by_setting[setting].setdefault(sample_id, row)
    all_ids = set.union(*(set(by_setting[setting]) for setting in SETTINGS))
    complete_ids = set.intersection(*(set(by_setting[setting]) for setting in SETTINGS))
    return {
        sample_id: {setting: by_setting[setting][sample_id] for setting in SETTINGS}
        for sample_id in sorted(complete_ids)
    }, source_count, len(all_ids)


def collect_train_ids(paths: list[Path]) -> tuple[set[str], dict[str, int]]:
    ids: set[str] = set()
    counts: dict[str, int] = {}
    for path in paths:
        rows = read_jsonl(path)
        found: set[str] = set()
        for row in rows:
            for candidate in (row.get("source_id"), row.get("id"), (row.get("metadata") or {}).get("source_id")):
                normalized = normalize_id(candidate, row if candidate == row.get("id") else None)
                if normalized:
                    found.add(normalized)
        ids.update(found)
        counts[path.as_posix()] = len(found)
    return ids, counts


def select_payload(eligible: list[str], requested: int, seed: int, pinned: list[str] | None = None) -> dict[str, Any]:
    pinned = [sample_id for sample_id in (pinned or []) if sample_id in set(eligible)]
    remainder = sorted(set(eligible) - set(pinned))
    random.Random(seed).shuffle(remainder)
    selected = (pinned + remainder)[:requested]
    warnings = []
    if len(selected) < requested:
        warnings.append(f"requested {requested} held-out IDs, but only {len(selected)} leakage-free aligned IDs are available")
    return {"ids": selected, "requested": requested, "selected": len(selected), "seed": seed, "warnings": warnings, "is_full_size": len(selected) == requested}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build(args: argparse.Namespace) -> dict[str, Any]:
    pool, source_normal_count, all_candidate_ids = aligned_rows([Path(item) for item in args.visual_control_dirs.split(",") if item], args.dataset)
    train_paths = sorted(Path("data/train").rglob("*.jsonl")) + sorted(Path("data/mining").rglob("*.jsonl"))
    train_ids, train_breakdown = collect_train_ids(train_paths)
    candidate_ids = sorted(pool)
    excluded = sorted(set(candidate_ids) & train_ids)
    eligible = sorted(set(candidate_ids) - train_ids)
    old_payload = json.loads(Path(args.previous_heldout).read_text(encoding="utf-8"))
    previous = [normalize_id(item) for item in (old_payload.get("ids") if isinstance(old_payload, dict) else old_payload)]
    previous_retained = [item for item in previous if item in eligible]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    for size in (100, 300, 500):
        payload = select_payload(eligible, size, args.seed, pinned=previous if size >= 100 else [])
        payload.update({"four_setting_complete": True, "leakage_after_filter": len(set(payload["ids"]) & train_ids)})
        (output / f"lingoqa_heldout_ids_{size}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payloads[str(size)] = payload
    merged_dir = Path(args.merged_pool_dir)
    for setting in SETTINGS:
        write_jsonl(merged_dir / f"{args.dataset}_strict_{setting}.jsonl", [pool[sample_id][setting] for sample_id in eligible])
    summary = {
        "dataset": args.dataset,
        "total_candidate_ids": len(candidate_ids),
        "source_normal_rows_before_merge": source_normal_count,
        "train_excluded_ids": len(excluded),
        "eligible_heldout_ids": len(eligible),
        "final_100_count": payloads["100"]["selected"],
        "final_300_count": payloads["300"]["selected"],
        "final_500_count": payloads["500"]["selected"],
        "leakage_after_filter": max(payload["leakage_after_filter"] for payload in payloads.values()),
        "four_setting_complete_rate": len(pool) / all_candidate_ids if all_candidate_ids else 0.0,
        "random_seed": args.seed,
        "stage4_5_previous_heldout_count": len(previous),
        "stage4_5_heldout_retained_count": len(previous_retained),
        "new_100_matches_stage4_5": payloads["100"]["ids"] == previous,
        "train_source_files": train_breakdown,
        "warnings": sorted(set(payloads["300"]["warnings"] + payloads["500"]["warnings"])),
        "merged_pool_dir": args.merged_pool_dir,
    }
    (output / "lingoqa_large_heldout_pool_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# LingoQA Larger Held-out Pool Summary", "",
        f"- Candidate IDs with four complete settings: {summary['total_candidate_ids']}",
        f"- Excluded by known training/mining sources: {summary['train_excluded_ids']}",
        f"- Eligible leakage-free IDs: {summary['eligible_heldout_ids']}",
        f"- Selected 100 / 300 / 500: {summary['final_100_count']} / {summary['final_300_count']} / {summary['final_500_count']}",
        f"- Leakage after filtering: {summary['leakage_after_filter']}",
        f"- Stage 4.5 held-out 100 preserved exactly: {'yes' if summary['new_100_matches_stage4_5'] else 'no'}", "",
        "## Conclusion", "",
        "The existing 300-item `eval_*` strict-control pool is covered by known training/mining sources and is excluded. The independent Stage 4.5 `test_*` pool contributes 100 valid IDs. Therefore a truthful 300/500 held-out evaluation cannot yet be run without acquiring or constructing new untouched LingoQA samples.",
    ]
    if summary["warnings"]:
        lines += ["", "## Warnings", *[f"- {warning}" for warning in summary["warnings"]]]
    (output / "lingoqa_large_heldout_pool_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build large leakage-filtered LingoQA held-out pools.")
    parser.add_argument("--dataset", default="lingoqa")
    parser.add_argument("--visual_control_dirs", default="data/processed/visual_control,data/processed/visual_control_heldout")
    parser.add_argument("--previous_heldout", default="outputs/final_report/stage4_5_heldout_ids_100.json")
    parser.add_argument("--merged_pool_dir", default="data/processed/visual_control_large_heldout")
    parser.add_argument("--output_dir", default="outputs/final_report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
