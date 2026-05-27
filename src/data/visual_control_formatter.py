"""Build strict visual-control inputs for LingoQA and DriveLM.

The formatter is intentionally offline and CPU-only. It never calls a model and
keeps strict visual-control samples free of perception JSON or raw object labels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETTINGS = ("normal", "text_only", "wrong_image", "blank_image")
CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
MODE_STRICT = "strict_visual"
MODE_AUGMENTED = "perception_augmented"


def read_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("answer") or answer.get("reason") or "")
    return str(answer or row.get("gold") or "")


def question_text(row: dict[str, Any]) -> str:
    return str(row.get("question") or row.get("instruction") or row.get("query") or "")


def external_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    ext = meta.get("external") if isinstance(meta.get("external"), dict) else {}
    return ext


def dataset_name(row: dict[str, Any], fallback: str) -> str:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    ext = external_meta(row)
    name = str(ext.get("benchmark_source") or meta.get("benchmark_source") or fallback).lower()
    if "lingo" in name:
        return "lingoqa"
    if "drive" in name:
        return "drivelm"
    return fallback


def image_paths(row: dict[str, Any]) -> list[str]:
    ext = external_meta(row)
    for key in ("image_paths", "original_image_paths", "ablated_image_paths"):
        value = ext.get(key)
        if isinstance(value, list) and value:
            return [str(x) for x in value if x is not None]
    value = row.get("image_paths")
    if isinstance(value, list) and value:
        return [str(x) for x in value if x is not None]
    image = row.get("image")
    return [str(image)] if image else []


def camera_from_path(path: str) -> str | None:
    for cam in sorted(CAMERA_ORDER, key=len, reverse=True):
        if cam in path:
            return cam
    return None


def camera_path_map(paths: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for idx, path in enumerate(paths):
        cam = camera_from_path(path)
        if cam:
            mapping[cam] = path
        elif idx < len(CAMERA_ORDER):
            mapping.setdefault(CAMERA_ORDER[idx], path)
    return mapping


def parse_question_cameras(question: str) -> list[str]:
    found: list[str] = []
    for cam in re.findall(r"<[^>]*?(CAM_[A-Z_]+)[^>]*?>", question.upper()):
        if cam in CAMERA_ORDER and cam not in found:
            found.append(cam)
    upper = question.upper()
    for cam in sorted(CAMERA_ORDER, key=len, reverse=True):
        if re.search(rf"\b{re.escape(cam)}\b", upper) and cam not in found:
            found.append(cam)
    return found or ["CAM_FRONT"]


def select_drivelm_images(question: str, paths: list[str]) -> tuple[list[str], list[str], list[str]]:
    mapping = camera_path_map(paths)
    requested = parse_question_cameras(question)
    selected: list[str] = []
    labels: list[str] = []
    for cam in requested:
        if cam in mapping:
            selected.append(mapping[cam])
            labels.append(f"[{cam}]")
    if not selected and paths:
        selected = [paths[0]]
        labels = [f"[{camera_from_path(paths[0]) or 'CAM_FRONT'}]"]
    return selected, labels, requested


def lingoqa_labels(paths: list[str]) -> list[str]:
    return [f"[Frame {idx}]" for idx, _ in enumerate(paths)]


def prompt_with_labels(question: str, labels: list[str]) -> str:
    return "\n".join([*labels, question]) if labels else question


def strict_metadata(row: dict[str, Any], dataset: str, setting: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    ext = external_meta(row)
    safe = {
        "source": meta.get("source"),
        "benchmark_source": dataset,
        "split": ext.get("split") or meta.get("clean_split"),
        "capability": ext.get("capability") or meta.get("capability"),
        "difficulty": ext.get("difficulty") or meta.get("difficulty"),
        "original_id": ext.get("original_id") or ext.get("question_id"),
        "setting": setting,
    }
    if extra:
        safe.update(extra)
    return {k: v for k, v in safe.items() if v not in (None, "")}


def maybe_augmented_metadata(row: dict[str, Any], base: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode != MODE_AUGMENTED:
        return base
    enriched = dict(base)
    if "perception" in row:
        enriched["perception"] = row.get("perception")
    return enriched


def blank_paths_for(dataset: str, count: int, blank_image: str | None = None) -> list[str]:
    placeholder = blank_image or f"outputs/cases/ablation_images/{dataset}_blank.jpg"
    return [placeholder for _ in range(max(1, count))]


def choose_wrong_paths(row_id: str, rows: list[dict[str, Any]], original_paths: list[str], dataset: str, rng: random.Random) -> list[str]:
    candidates = [r for r in rows if str(r.get("id")) != row_id]
    rng.shuffle(candidates)
    original_tuple = tuple(original_paths)
    for candidate in candidates:
        paths = image_paths(candidate)
        if paths and tuple(paths) != original_tuple:
            return paths
    if original_paths:
        return [f"wrong_image_placeholder::{dataset}::{idx}" for idx, _ in enumerate(original_paths)]
    return [f"wrong_image_placeholder::{dataset}::0"]


def format_sample(row: dict[str, Any], dataset: str, setting: str, paths: list[str], labels: list[str], mode: str, metadata_extra: dict[str, Any] | None = None) -> dict[str, Any]:
    question = question_text(row)
    meta = maybe_augmented_metadata(row, strict_metadata(row, dataset, setting, metadata_extra), mode)
    return {
        "id": str(row.get("id") or meta.get("original_id") or ""),
        "dataset": dataset,
        "mode": mode,
        "setting": setting,
        "question": question,
        "gold": answer_text(row),
        "image_paths": list(paths),
        "image_labels": list(labels),
        "prompt": prompt_with_labels(question, labels),
        "metadata": meta,
    }


def build_variants(row: dict[str, Any], all_rows: list[dict[str, Any]], dataset: str, mode: str = MODE_STRICT, rng: random.Random | None = None, blank_image: str | None = None) -> dict[str, dict[str, Any]]:
    if mode not in {MODE_STRICT, MODE_AUGMENTED}:
        raise ValueError(f"unsupported mode: {mode}")
    rng = rng or random.Random(0)
    original = image_paths(row)
    question = question_text(row)
    row_id = str(row.get("id") or "")
    if dataset == "drivelm":
        normal_paths, normal_labels, requested = select_drivelm_images(question, original)
        wrong_all = choose_wrong_paths(row_id, all_rows, original, dataset, rng)
        wrong_paths, wrong_labels, _ = select_drivelm_images(question, wrong_all)
        blank_paths, blank_labels, _ = select_drivelm_images(question, blank_paths_for(dataset, len(original) or 1, blank_image))
        meta_extra = {"selected_cameras": [label.strip("[]") for label in normal_labels], "requested_cameras": requested}
    else:
        normal_paths = original
        normal_labels = lingoqa_labels(normal_paths)
        wrong_paths = choose_wrong_paths(row_id, all_rows, original, dataset, rng)
        wrong_labels = lingoqa_labels(wrong_paths)
        blank_paths = blank_paths_for(dataset, len(original) or 1, blank_image)
        blank_labels = lingoqa_labels(blank_paths)
        meta_extra = {"frame_count": len(normal_paths)}
    return {
        "normal": format_sample(row, dataset, "normal", normal_paths, normal_labels, mode, meta_extra),
        "text_only": format_sample(row, dataset, "text_only", [], [], mode, meta_extra),
        "wrong_image": format_sample(row, dataset, "wrong_image", wrong_paths, wrong_labels, mode, meta_extra),
        "blank_image": format_sample(row, dataset, "blank_image", blank_paths, blank_labels, mode, meta_extra),
    }


def build_dataset(rows: list[dict[str, Any]], dataset: str, mode: str, seed: int, blank_image: str | None = None) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    outputs = {setting: [] for setting in SETTINGS}
    for row in rows:
        variants = build_variants(row, rows, dataset=dataset, mode=mode, rng=rng, blank_image=blank_image)
        for setting in SETTINGS:
            outputs[setting].append(variants[setting])
    return outputs


def output_stem(dataset: str, mode: str, setting: str) -> str:
    mode_name = "strict" if mode == MODE_STRICT else "perception_augmented"
    return f"{dataset}_{mode_name}_{setting}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CPU-only visual-control JSONL files.")
    parser.add_argument("--input_lingoqa", default="data/processed/lingoqa_clean_v2_train.jsonl")
    parser.add_argument("--input_drivelm", default="data/processed/drivelm_train_scene.jsonl")
    parser.add_argument("--output_dir", default="data/processed/visual_control")
    parser.add_argument("--max_lingoqa", type=int, default=0)
    parser.add_argument("--max_drivelm", type=int, default=0)
    parser.add_argument("--mode", choices=[MODE_STRICT, MODE_AUGMENTED], default=MODE_STRICT)
    parser.add_argument("--blank_image", default="")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    summary: dict[str, Any] = {"mode": args.mode, "dry_run": bool(args.dry_run), "outputs": {}}
    specs = [("lingoqa", Path(args.input_lingoqa), args.max_lingoqa), ("drivelm", Path(args.input_drivelm), args.max_drivelm)]
    for dataset, path, limit in specs:
        if not path.exists():
            summary["outputs"][dataset] = {"skipped": f"missing input: {path}"}
            continue
        effective_limit = limit
        if args.dry_run and (effective_limit <= 0 or effective_limit > 8):
            effective_limit = 8
        rows = read_jsonl(path, effective_limit)
        outputs = build_dataset(rows, dataset, args.mode, args.seed, blank_image=args.blank_image or None)
        summary["outputs"][dataset] = {}
        for setting, setting_rows in outputs.items():
            out_path = output_dir / output_stem(dataset, args.mode, setting)
            count = write_jsonl(out_path, setting_rows)
            summary["outputs"][dataset][setting] = {"path": out_path.as_posix(), "count": count}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
