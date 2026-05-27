"""Prepare a DriveLM OOD strict visual-control pool without model inference."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")
CAMERAS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
BLANK_MARKERS = ("blank", "placeholder", "empty")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_label(label: Any) -> str:
    return str(label or "").strip().strip("[]")


def classify_question(question: str) -> str:
    value = question.lower()
    if re.search(r"<c\d+,\s*cam_", value, re.I):
        return "object_token"
    if re.search(r"cam_(?:front|back)", value, re.I):
        return "camera_specific"
    if any(word in value for word in ("left", "right", "front", "behind", "ahead", "near", "relative")):
        return "spatial_relation"
    if any(word in value for word in ("how many", "number of", "count")):
        return "counting"
    if value.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ")):
        return "yes_no"
    if any(word in value for word in ("action", "should", "why", "dangerous", "ego vehicle")):
        return "action_reasoning"
    return "other"


def has_perception_json_leak(row: dict[str, Any]) -> bool:
    exposed = " ".join(str(row.get(key, "")) for key in ("prompt", "question", "gold"))
    return bool(re.search(r'["\']?(?:perception|objects|bbox|coordinates)["\']?\s*:', exposed, re.I))


def normalize_source_id(value: Any, row: dict[str, Any] | None = None) -> str:
    direct = (row or {}).get("source_id") or ((row or {}).get("metadata") or {}).get("source_id")
    if direct:
        return str(direct)
    return re.sub(r"_(?:normal_replay|normal_visual_qa|blank_calibration|wrong_image_calibration|text_only_calibration|spatial_normal_qa|extra_normal_fill)$", "", str(value or ""))


def validate_group(group: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    normal = group["normal"]
    if any(str(group[setting].get("mode", "")) != "strict_visual" for setting in SETTINGS):
        errors.append("mode_not_strict_visual")
    if any(has_perception_json_leak(group[setting]) for setting in SETTINGS):
        errors.append("perception_json_leak")
    if group["text_only"].get("image_paths"):
        errors.append("text_only_has_images")
    if group["wrong_image"].get("image_paths") == normal.get("image_paths"):
        errors.append("wrong_image_matches_normal")
    blank_paths = [str(path).lower() for path in group["blank_image"].get("image_paths", [])]
    if not blank_paths or any(not any(marker in path for marker in BLANK_MARKERS) for path in blank_paths):
        errors.append("blank_image_not_placeholder")
    labels = normal.get("image_labels") or []
    if len(normal.get("image_paths") or []) > 1 and not labels:
        errors.append("multi_camera_missing_labels")
    return errors


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.visual_control_dir)
    rows = {setting: read_jsonl(root / f"drivelm_strict_{setting}.jsonl") for setting in SETTINGS}
    maps = {setting: {str(row.get("id")): row for row in rows[setting]} for setting in SETTINGS}
    complete = sorted(set.intersection(*(set(maps[setting]) for setting in SETTINGS)))
    invalid: dict[str, list[str]] = {}
    eligible: list[str] = []
    for sample_id in complete:
        errors = validate_group({setting: maps[setting][sample_id] for setting in SETTINGS})
        if errors:
            invalid[sample_id] = errors
        else:
            eligible.append(sample_id)
    random.Random(args.seed).shuffle(eligible)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for size in (100, 300):
        selected = eligible[:size]
        payload = {
            "ids": selected, "requested": size, "selected": len(selected), "seed": args.seed,
            "is_full_size": len(selected) == size,
            "warnings": [] if len(selected) == size else [f"only {len(selected)} valid OOD IDs are available for requested {size}"],
        }
        (output / f"drivelm_ood_ids_{size}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prepared = Path(args.prepared_pool_dir)
    for setting in SETTINGS:
        prepared.mkdir(parents=True, exist_ok=True)
        (prepared / f"drivelm_strict_{setting}.jsonl").write_text(
            "".join(json.dumps(maps[setting][sample_id], ensure_ascii=False) + "\n" for sample_id in eligible),
            encoding="utf-8",
        )
    label_coverage = Counter()
    multi_count = 0
    multi_full_rig_count = 0
    types = Counter()
    camera_aware = 0
    for sample_id in eligible:
        row = maps["normal"][sample_id]
        labels = {normalize_label(item) for item in row.get("image_labels", [])}
        label_coverage.update(labels)
        if len(row.get("image_paths") or []) > 1:
            multi_count += 1
            multi_full_rig_count += set(CAMERAS).issubset(labels)
        q = str(row.get("question", ""))
        types[classify_question(q)] += 1
        camera_aware += bool(re.search(r"CAM_BACK|CAM_FRONT_LEFT|<c\d+,\s*CAM_", q, re.I))
    prior_overlap = 0
    for path in (Path("data/train/sft_v3/drivelm_sft_v3.jsonl"), Path("data/train/sft_v3_r2/drivelm_sft_v3_r2.jsonl")):
        if path.exists():
            prior_ids = {normalize_source_id(row.get("id"), row) for row in read_jsonl(path)}
            prior_overlap = max(prior_overlap, len(prior_ids & set(eligible)))
    required_missing_global = [camera for camera in CAMERAS if label_coverage[camera] == 0]
    warnings = []
    if required_missing_global:
        warnings.append(f"camera labels absent across OOD pool: {required_missing_global}")
    if multi_count and multi_full_rig_count < multi_count:
        warnings.append(f"{multi_count - multi_full_rig_count}/{multi_count} multi-image samples do not contain all six camera labels; labels represent selected views only")
    if prior_overlap:
        warnings.append(f"DriveLM rows overlap earlier mixed SFT branches ({prior_overlap} IDs), but the selected final r3 adapter was trained on LingoQA only; treat DriveLM as r3 cross-dataset diagnosis")
    summary = {
        "dataset": "drivelm",
        "role": "OOD diagnostic evaluation for final LingoQA-trained SFT-v3-r3 checkpoint",
        "four_setting_candidate_ids": len(complete),
        "eligible_ood_ids": len(eligible),
        "invalid_ids": len(invalid),
        "invalid_reason_counts": dict(Counter(reason for reasons in invalid.values() for reason in reasons)),
        "four_setting_complete": len(complete) == min(len(rows[setting]) for setting in SETTINGS),
        "selected_100": min(100, len(eligible)),
        "selected_300": min(300, len(eligible)),
        "camera_label_coverage": dict(label_coverage),
        "multi_image_samples": multi_count,
        "multi_image_full_six_camera_samples": multi_full_rig_count,
        "camera_aware_coverage": camera_aware,
        "camera_aware_rate": camera_aware / len(eligible) if eligible else 0.0,
        "question_type_distribution": dict(types),
        "prior_mixed_branch_overlap_count": prior_overlap,
        "warnings": warnings,
        "seed": args.seed,
        "prepared_pool_dir": args.prepared_pool_dir,
    }
    (output / "drivelm_ood_pool_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DriveLM OOD Strict Visual-Control Pool", "",
        "DriveLM is used as an OOD diagnostic set for the final LingoQA-trained `SFT-v3-r3` checkpoint. An OOD score does not have to improve; it diagnoses cross-dataset generalization and multi-camera alignment.", "",
        f"- Four-setting complete candidates: {summary['four_setting_candidate_ids']}",
        f"- Eligible OOD IDs after structural validation: {summary['eligible_ood_ids']}",
        f"- Selected 100 / 300: {summary['selected_100']} / {summary['selected_300']}",
        f"- Invalid samples: {summary['invalid_ids']}",
        f"- Camera-aware questions: {summary['camera_aware_coverage']} ({summary['camera_aware_rate']:.2%})",
        "", "## Capability / Question Type", "",
        *[f"- {kind}: {count}" for kind, count in sorted(types.items())],
        "", "## Camera Labels", "",
        *[f"- {camera}: {label_coverage[camera]}" for camera in CAMERAS],
    ]
    if warnings:
        lines += ["", "## Warnings", *[f"- {warning}" for warning in warnings]]
    (output / "drivelm_ood_pool_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DriveLM OOD strict visual-control evaluation pool.")
    parser.add_argument("--visual_control_dir", default="data/processed/visual_control")
    parser.add_argument("--prepared_pool_dir", default="data/processed/visual_control_drivelm_ood")
    parser.add_argument("--output_dir", default="outputs/final_report")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
