
"""Build SFT-v3-r2 data with guarded normal/control ratios.

This builder keeps the Stage 2 SFT-v3 artifacts intact and writes a new r2
training set focused on recovering normal visual QA ability while preserving
control calibration.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.visual_control_formatter import write_jsonl

SETTINGS = ("normal", "blank_image", "wrong_image", "text_only")
CONTROL_TYPES = {"blank_refusal", "wrong_image_caution", "text_only_caution"}
BAD_FIELDS = {"references", "category", "subcategory", "debug"}
REFUSAL_RE = re.compile(
    r"\u65e0\u6cd5\u5224\u65ad|\u65e0\u6cd5\u786e\u5b9a|\u7f3a\u5c11\u56fe\u50cf\u4fe1\u606f|\u56fe\u50cf\u4fe1\u606f\u4e0d\u8db3|cannot determine|not enough (?:visual )?information|unable to determine",
    re.I,
)
PATH_RE = re.compile(r"(?:^|[\s\"'])/?(?:root/|data/|outputs/|checkpoints/|src/)[^\s\"']+", re.I)
SPATIAL_RE = re.compile(r"left|right|front|back|lane|traffic light|pedestrian|vehicle|car|behind|ahead|CAM_FRONT|CAM_BACK", re.I)

NORMAL_REASON = "The answer is based on the visible driving scene."
NORMAL_SHORT_REASON = "Visible evidence supports this answer."
BLANK_ANSWER = "\u5f53\u524d\u56fe\u50cf\u4fe1\u606f\u4e0d\u8db3\uff0c\u65e0\u6cd5\u786e\u5b9a\u3002"
BLANK_REASON = "The blank image does not provide usable visual evidence."
TEXT_ONLY_ANSWER = "\u7f3a\u5c11\u56fe\u50cf\u4fe1\u606f\uff0c\u65e0\u6cd5\u53ef\u9760\u5224\u65ad\u3002"
TEXT_ONLY_REASON = "No image is provided, so visual evidence is unavailable."
WRONG_IMAGE_ANSWER = "The provided image does not reliably support answering the original question."
WRONG_IMAGE_REASON = "The visual evidence may not correspond to the original scene."

TARGET_RATIOS = {
    "normal_visual_qa": 0.58,
    "blank_refusal": 0.10,
    "wrong_image_caution": 0.10,
    "text_only_caution": 0.08,
    "spatial_hard_negative": 0.14,
}


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assistant(answer: str, reason: str) -> dict[str, str]:
    return {"answer": str(answer).strip(), "reason": str(reason).strip()}


def json_content(obj: dict[str, str]) -> str:
    return json.dumps({"answer": obj.get("answer", ""), "reason": obj.get("reason", "")}, ensure_ascii=False)


def has_bad_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(k) in BAD_FIELDS or has_bad_field(v) for k, v in value.items())
    if isinstance(value, list):
        return any(has_bad_field(v) for v in value)
    return False


def leaks_path(text: str) -> bool:
    return bool(PATH_RE.search(text or ""))


def is_refusal_text(text: str) -> bool:
    return bool(REFUSAL_RE.search(text or ""))


def clean_gold(gold: Any) -> str:
    value = str(gold or "").strip()
    return value if value else "unknown"


def normal_anchor_cleaner(sample: dict[str, Any]) -> dict[str, str] | None:
    gold = clean_gold(sample.get("gold"))
    if is_refusal_text(gold) or leaks_path(gold):
        return None
    reason = NORMAL_SHORT_REASON if len(gold.split()) <= 5 else NORMAL_REASON
    return assistant(gold, reason)


def make_sft_row(sample: dict[str, Any], sample_type: str, out: dict[str, str], source_setting: str) -> dict[str, Any]:
    source_id = str(sample.get("id") or "unknown")
    prompt = str(sample.get("prompt") or sample.get("question") or "")
    return {
        "id": f"{source_id}_{sample_type}",
        "dataset": sample.get("dataset", ""),
        "sample_type": sample_type,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json_content(out)},
        ],
        "image_paths": sample.get("image_paths", []),
        "image_labels": sample.get("image_labels", []),
        "prompt": prompt,
        "assistant": out,
        "metadata": {
            "source_id": source_id,
            "setting": source_setting,
            "r2_builder": True,
        },
    }


def read_groups(visual_control_dir: Path, dataset: str, limit_cases: int = 0) -> list[dict[str, dict[str, Any]]]:
    maps: dict[str, dict[str, dict[str, Any]]] = {setting: {} for setting in SETTINGS}
    order: list[str] = []
    for setting in SETTINGS:
        path = visual_control_dir / f"{dataset}_strict_{setting}.jsonl"
        for row in read_jsonl(path):
            sid = str(row.get("id") or "")
            if not sid:
                continue
            maps[setting][sid] = row
            if setting == "normal":
                order.append(sid)
    common = set(maps["normal"])
    for setting in SETTINGS[1:]:
        common &= set(maps[setting])
    groups: list[dict[str, dict[str, Any]]] = []
    for sid in order:
        if sid in common:
            groups.append({setting: maps[setting][sid] for setting in SETTINGS})
        if limit_cases and len(groups) >= limit_cases:
            break
    return groups


def build_candidates(groups: list[dict[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        normal = group["normal"]
        normal_out = normal_anchor_cleaner(normal)
        if normal_out:
            candidates["normal_visual_qa"].append(make_sft_row(normal, "normal_visual_qa", normal_out, "normal"))
        candidates["blank_refusal"].append(make_sft_row(group["blank_image"], "blank_refusal", assistant(BLANK_ANSWER, BLANK_REASON), "blank_image"))
        candidates["wrong_image_caution"].append(make_sft_row(group["wrong_image"], "wrong_image_caution", assistant(WRONG_IMAGE_ANSWER, WRONG_IMAGE_REASON), "wrong_image"))
        candidates["text_only_caution"].append(make_sft_row(group["text_only"], "text_only_caution", assistant(TEXT_ONLY_ANSWER, TEXT_ONLY_REASON), "text_only"))
        question = str(normal.get("question") or normal.get("prompt") or "")
        if SPATIAL_RE.search(question):
            spatial_out = normal_anchor_cleaner(normal)
            if spatial_out:
                candidates["spatial_hard_negative"].append(make_sft_row(normal, "spatial_hard_negative", spatial_out, "normal"))
    return candidates


def take_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = list(rows)
    rng.shuffle(rows)
    return rows[: max(0, min(count, len(rows)))]


def sample_r2(candidates: dict[str, list[dict[str, Any]]], seed: int, target_ratios: dict[str, float] | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    ratios = target_ratios or TARGET_RATIOS
    rng = random.Random(seed)
    warnings: list[str] = []
    normal_available = len(candidates.get("normal_visual_qa", []))
    if normal_available == 0:
        raise ValueError("No clean normal_visual_qa candidates available.")
    target_total = max(normal_available, round(normal_available / ratios["normal_visual_qa"]))
    target_counts = {k: int(round(target_total * v)) for k, v in ratios.items()}
    target_counts["normal_visual_qa"] = normal_available
    # Keep total stable after rounding by adjusting normal, never adding more control.
    non_normal = sum(v for k, v in target_counts.items() if k != "normal_visual_qa")
    if normal_available + non_normal > target_total:
        overflow = normal_available + non_normal - target_total
        for key in ["spatial_hard_negative", "text_only_caution", "blank_refusal", "wrong_image_caution"]:
            drop = min(overflow, target_counts.get(key, 0))
            target_counts[key] -= drop
            overflow -= drop
            if overflow <= 0:
                break
    out: list[dict[str, Any]] = []
    for sample_type in ["normal_visual_qa", "blank_refusal", "wrong_image_caution", "text_only_caution", "spatial_hard_negative"]:
        available = candidates.get(sample_type, [])
        want = target_counts.get(sample_type, 0)
        got = take_rows(available, want, rng)
        if len(got) < want:
            warnings.append(f"{sample_type}: requested {want}, available {len(got)}; deficit assigned to normal by ratio effect")
        out.extend(got)
    rng.shuffle(out)
    return out, warnings


def parse_assistant(row: dict[str, Any]) -> dict[str, str] | None:
    obj = row.get("assistant")
    if isinstance(obj, dict):
        return {"answer": str(obj.get("answer", "")), "reason": str(obj.get("reason", ""))}
    return None


def compute_stats(rows: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    total = len(rows)
    type_counts = Counter(str(r.get("sample_type") or "unknown") for r in rows)
    dataset_counts = Counter(str(r.get("dataset") or "unknown") for r in rows)
    answer_lengths: list[int] = []
    reason_lengths: list[int] = []
    invalid_outputs = 0
    metadata_leaks = 0
    normal_refusals = 0
    refusal_answers = 0
    control_count = sum(type_counts[t] for t in CONTROL_TYPES)
    for row in rows:
        out = parse_assistant(row)
        if not out or set(out.keys()) != {"answer", "reason"}:
            invalid_outputs += 1
            continue
        answer = out["answer"]
        reason = out["reason"]
        answer_lengths.append(len(answer.split()) if re.search(r"[A-Za-z]", answer) else len(answer))
        reason_lengths.append(len(reason.split()) if re.search(r"[A-Za-z]", reason) else len(reason))
        text = json.dumps(out, ensure_ascii=False)
        if has_bad_field(out) or leaks_path(text):
            metadata_leaks += 1
        if row.get("sample_type") == "normal_visual_qa" and is_refusal_text(text):
            normal_refusals += 1
        if row.get("sample_type") in CONTROL_TYPES and is_refusal_text(answer):
            refusal_answers += 1
    normal = type_counts.get("normal_visual_qa", 0)
    spatial = type_counts.get("spatial_hard_negative", 0)
    type_ratios = {k: v / total for k, v in type_counts.items()} if total else {}
    refusal_same_ratio = refusal_answers / control_count if control_count else 0.0
    guard_errors: list[str] = []
    if total == 0:
        guard_errors.append("empty dataset")
    if total and normal / total < 0.50:
        guard_errors.append("normal_visual_qa ratio below 0.50")
    if total and control_count / total > 0.38:
        guard_errors.append("control/refusal ratio above 0.38")
    if normal_refusals:
        warnings.append(f"normal_visual_qa contains {normal_refusals} refusal-like outputs")
    if refusal_same_ratio > 0.90:
        warnings.append(f"control refusal-like answer ratio is high: {refusal_same_ratio:.3f}")
    if invalid_outputs:
        guard_errors.append(f"invalid assistant outputs: {invalid_outputs}")
    if metadata_leaks:
        guard_errors.append(f"assistant metadata/path leaks: {metadata_leaks}")
    return {
        "total": total,
        "sample_type_counts": dict(type_counts),
        "sample_type_ratios": type_ratios,
        "dataset_counts": dict(dataset_counts),
        "normal_ratio": normal / total if total else 0.0,
        "control_ratio": control_count / total if total else 0.0,
        "spatial_ratio": spatial / total if total else 0.0,
        "refusal_ratio": (type_counts.get("blank_refusal", 0) + type_counts.get("text_only_caution", 0)) / total if total else 0.0,
        "control_refusal_answer_ratio": refusal_same_ratio,
        "normal_refusal_count": normal_refusals,
        "avg_answer_length": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
        "avg_reason_length": sum(reason_lengths) / len(reason_lengths) if reason_lengths else 0.0,
        "ratio_guard_passed": not guard_errors,
        "guard_errors": guard_errors,
        "warnings": warnings,
    }


def build_dataset(visual_control_dir: Path, dataset: str, seed: int, max_cases: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = read_groups(visual_control_dir, dataset, max_cases)
    candidates = build_candidates(groups)
    rows, warnings = sample_r2(candidates, seed)
    stats = compute_stats(rows, warnings)
    return rows, stats


def mix_datasets(lingoqa_rows: list[dict[str, Any]], drivelm_rows: list[dict[str, Any]], seed: int, lingo_ratio: float = 0.70) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    lingo = list(lingoqa_rows)
    drive = list(drivelm_rows)
    rng.shuffle(lingo)
    rng.shuffle(drive)
    if not drive:
        mixed = lingo
    else:
        target_drive = min(len(drive), round(len(lingo) * (1 - lingo_ratio) / lingo_ratio))
        mixed = lingo + drive[:target_drive]
    rng.shuffle(mixed)
    return mixed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build guarded SFT-v3-r2 data.")
    p.add_argument("--visual_control_dir", default="data/processed/visual_control")
    p.add_argument("--output_dir", default="data/train/sft_v3_r2")
    p.add_argument("--max_lingoqa_cases", type=int, default=0)
    p.add_argument("--max_drivelm_cases", type=int, default=0)
    p.add_argument("--skip_drivelm", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    max_lingo = args.max_lingoqa_cases or (24 if args.dry_run else 0)
    max_drive = args.max_drivelm_cases or (24 if args.dry_run else 0)
    out_dir = Path(args.output_dir)
    lingoqa_rows, lingoqa_stats = build_dataset(Path(args.visual_control_dir), "lingoqa", args.seed, max_lingo)
    drivelm_rows: list[dict[str, Any]] = []
    drivelm_stats: dict[str, Any] = {"total": 0, "warnings": ["DriveLM skipped"]}
    if not args.skip_drivelm:
        drivelm_rows, drivelm_stats = build_dataset(Path(args.visual_control_dir), "drivelm", args.seed + 1, max_drive)
    mixed = mix_datasets(lingoqa_rows, drivelm_rows, args.seed)
    mixed_stats = compute_stats(mixed, [])
    write_jsonl(out_dir / "lingoqa_sft_v3_r2.jsonl", lingoqa_rows)
    write_jsonl(out_dir / "drivelm_sft_v3_r2.jsonl", drivelm_rows)
    write_jsonl(out_dir / "mixed_sft_v3_r2.jsonl", mixed)
    write_jsonl(out_dir / "sft_v3_r2_preview.jsonl", lingoqa_rows[:20])
    stats = {
        "lingoqa": lingoqa_stats,
        "drivelm": drivelm_stats,
        "mixed": mixed_stats,
    }
    write_json(out_dir / "sft_v3_r2_stats.json", stats)
    if not lingoqa_stats.get("ratio_guard_passed"):
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    print(json.dumps({"output_dir": out_dir.as_posix(), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
