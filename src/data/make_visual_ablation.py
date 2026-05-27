"""Create visual-ablation JSONL variants for grounding checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_blank_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:
        raise RuntimeError("blank_image ablation requires pillow") from exc
    image = Image.new("RGB", size, color=(128, 128, 128))
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "visual ablation blank image", fill=(255, 255, 255))
    image.save(path)


def row_image_paths(row: dict[str, Any]) -> list[str]:
    external = row.get("meta", {}).get("external", {})
    paths = external.get("image_paths", []) if isinstance(external, dict) else []
    if isinstance(paths, list) and paths:
        return [str(path) for path in paths if str(path)]
    image = str(row.get("image", ""))
    return [image] if image else []


def external_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    external = meta.get("external", {}) if isinstance(meta.get("external"), dict) else {}
    return external


def row_scene_token(row: dict[str, Any]) -> str:
    external = external_meta(row)
    return str(external.get("scene_token") or row.get("scene_token") or "")


def row_sample_token(row: dict[str, Any]) -> str:
    external = external_meta(row)
    return str(external.get("sample_token") or row.get("sample_token") or row.get("id") or "")


def row_capability(row: dict[str, Any]) -> str:
    meta = row.get("meta", {}) if isinstance(row.get("meta"), dict) else {}
    external = external_meta(row)
    return str(meta.get("capability") or external.get("capability") or "")


def row_category(row: dict[str, Any]) -> str:
    gold = row.get("answer") if isinstance(row.get("answer"), dict) else row.get("gold", {})
    external = external_meta(row)
    return str((gold or {}).get("category") or external.get("category") or "")


def row_answer_text(row: dict[str, Any]) -> str:
    answer = row.get("answer") if isinstance(row.get("answer"), dict) else row.get("gold", {})
    if not isinstance(answer, dict):
        return str(answer or "").lower().strip()
    return str(answer.get("answer") or answer.get("reason") or "").lower().strip()


def is_same_scene_or_sample(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_scene = row_scene_token(left)
    right_scene = row_scene_token(right)
    if left_scene and right_scene and left_scene == right_scene:
        return True
    left_sample = row_sample_token(left)
    right_sample = row_sample_token(right)
    return bool(left_sample and right_sample and left_sample == right_sample)


def choose_wrong_index(rows: list[dict[str, Any]], idx: int, strategy: str, rng: random.Random) -> int:
    if len(rows) <= 1:
        return idx
    if strategy == "next":
        return (idx + 1) % len(rows)

    row = rows[idx]
    candidates = [i for i, candidate in enumerate(rows) if i != idx and not is_same_scene_or_sample(row, candidate)]
    if not candidates:
        candidates = [i for i in range(len(rows)) if i != idx]
    if strategy == "random":
        return rng.choice(candidates)

    capability = row_capability(row)
    category = row_category(row)
    answer_text = row_answer_text(row)

    def score(candidate_idx: int) -> tuple[int, float]:
        candidate = rows[candidate_idx]
        value = 0
        if capability and row_capability(candidate) == capability:
            value += 6
        if category and row_category(candidate) == category:
            value += 4
        if answer_text and row_answer_text(candidate) and row_answer_text(candidate) != answer_text:
            value += 3
        if row_scene_token(row) and row_scene_token(candidate) and row_scene_token(row) != row_scene_token(candidate):
            value += 2
        return value, rng.random()

    if strategy == "hard_scene":
        return max(candidates, key=score)
    raise ValueError(f"unsupported wrong_strategy: {strategy}")


def build_variant(
    rows: list[dict[str, Any]],
    mode: str,
    blank_image_path: Path,
    blank_size: tuple[int, int],
    wrong_strategy: str,
    seed: int,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    rng = random.Random(seed)
    if mode == "wrong_image":
        images = [row_image_paths(row) for row in rows]
    elif mode == "blank_image":
        create_blank_image(blank_image_path, blank_size)
        max_frames = max((len(row_image_paths(row)) for row in rows), default=1)
        images = [[blank_image_path.as_posix()] * max_frames for _ in rows]
    else:
        raise ValueError(f"unsupported mode: {mode}")

    variant: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        new_row = json.loads(json.dumps(row, ensure_ascii=False))
        if mode == "wrong_image":
            replacement_idx = choose_wrong_index(rows, idx, wrong_strategy, rng)
            replacement_paths = images[replacement_idx]
            replacement_row = rows[replacement_idx]
        else:
            replacement_paths = images[idx]
            replacement_idx = idx
            replacement_row = row
        new_row["image"] = replacement_paths[0] if replacement_paths else ""
        meta = new_row.setdefault("meta", {})
        external = meta.setdefault("external", {})
        meta["visual_ablation"] = mode
        meta["visual_ablation_strategy"] = wrong_strategy if mode == "wrong_image" else "blank_image"
        external["original_image"] = row.get("image", "")
        external["original_image_paths"] = row_image_paths(row)
        external["original_scene_token"] = row_scene_token(row)
        external["original_sample_token"] = row_sample_token(row)
        external["ablated_image"] = new_row.get("image", "")
        external["image_paths"] = replacement_paths
        external["ablated_image_paths"] = replacement_paths
        external["ablated_source_id"] = replacement_row.get("id", "")
        external["ablated_scene_token"] = row_scene_token(replacement_row)
        external["ablated_sample_token"] = row_sample_token(replacement_row)
        variant.append(new_row)
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(description="Create wrong-image or blank-image visual ablation datasets.")
    parser.add_argument("--input", default="data/processed/drivemind_intelli_sample.jsonl")
    parser.add_argument("--output", default="data/processed/drivemind_intelli_sample_wrong_image.jsonl")
    parser.add_argument("--mode", choices=["wrong_image", "blank_image"], required=True)
    parser.add_argument("--blank_image_path", default="outputs/cases/ablation_images/blank.jpg")
    parser.add_argument("--blank_width", type=int, default=640)
    parser.add_argument("--blank_height", type=int, default=360)
    parser.add_argument(
        "--wrong_strategy",
        choices=["next", "random", "hard_scene"],
        default="next",
        help="How to choose replacement frames for wrong-image ablations.",
    )
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    variant = build_variant(
        rows,
        mode=args.mode,
        blank_image_path=Path(args.blank_image_path),
        blank_size=(args.blank_width, args.blank_height),
        wrong_strategy=args.wrong_strategy,
        seed=args.seed,
    )
    write_jsonl(variant, Path(args.output))
    print(f"wrote {len(variant)} {args.mode} samples to {args.output}")


if __name__ == "__main__":
    main()
